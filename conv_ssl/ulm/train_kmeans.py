from argparse import ArgumentParser
from tqdm import tqdm
from os.path import join, exists, basename
from os import makedirs, environ
import time
from halo import Halo  # spinner
from omegaconf import OmegaConf

# No warnings that dataset is already extracted
environ["DATASETS_VERBOSITY"] = "error"

import torch
import einops

from datasets_turntaking import DialogAudioDM

from conv_ssl.models.encoder import Encoder
from conv_ssl.models.kmean import KMeanEmbedding, KmeanSKLearn
from conv_ssl.utils import count_parameters


""" 
Build Dataset
python conv_ssl/train_kmeans.py \
        --savepath dataset_units \
        --model hubert \
        --kmeans_max_feature_memory 10 \
        --extract_dataset # flag to also precompute units over the datasets
"""


def get_memory(x, size_type="gb"):
    """return size in bytes"""
    m = x.element_size() * x.nelement()
    if size_type == "gb":
        m /= 1e9
    elif size_type == "mb":
        m /= 1e6
    elif size_type == "kb":
        m /= 1e3
    return m


class SegmentDatasetBuilder:
    def __init__(self, args) -> None:
        self.args = args
        self.encoder = None
        self.model_config = Encoder.load_config(args.conf)
        self.dm = self._load_datamodule()
        self.paths = self._filepaths()
        makedirs(self.paths["root"], exist_ok=True)
        makedirs(self.paths["train_files_root"], exist_ok=True)
        makedirs(self.paths["val_files_root"], exist_ok=True)
        makedirs(self.paths["val_subset"], exist_ok=True)

    def _filepaths(self):
        root = join(args.savepath, self.model_config["encoder"]["type"])

        k_training_features = (
            args.k_training_features_path
            if args.k_training_features_path
            else join(
                root,
                f"{self.model_config['encoder']['type']}_k_{self.model_config['quantizer']['n_codes']}_training_features.pt",
            )
        )

        k_vectors = (
            self.args.k_vectors_path
            if self.args.k_vectors_path
            else join(root, "k_vectors.pt")
        )

        return {
            "root": root,
            "k_vectors": k_vectors,
            "k_training_features": k_training_features,
            "train_files_root": join(root, "train"),
            "val_files_root": join(root, "val"),
            "val_subset": join(root, "val_audio"),
            "conf_dm": join(root, "conf_dm.yaml"),
            "conf_model": join(root, "conf_model.yaml"),
        }

    def save_configs(self):
        if self.encoder is not None:
            OmegaConf.save(self.encoder.conf, self.paths["conf_model"])
            print("Saved Model config -> ", self.paths["conf_model"])

        # DM conf
        OmegaConf.save(self.dm.config, self.paths["conf_dm"])
        print("Saved DM config -> ", self.paths["conf_dm"])

    def _load_model(self):
        spinner = Halo(text="Loading Model", spinner="dots")
        spinner.start()

        model = Encoder(self.model_config)
        model.device = "cpu"
        if torch.cuda.is_available():
            model = model.to("cuda")
            model.device = "cuda"
        n = count_parameters(model, as_string=True, learnable=False)
        spinner.succeed(f"Model {model.name}: {n} params")
        return model

    def _load_datamodule(self):
        data_conf = DialogAudioDM.load_config(path=self.args.data_conf, args=self.args)
        data_conf["dataset"]["type"] = "sliding"
        data_conf["dataset"]["audio_overlap"] = 9  # 10 second windows
        dm = DialogAudioDM(
            datasets=data_conf["dataset"]["datasets"],
            type=data_conf["dataset"]["type"],
            audio_duration=data_conf["dataset"]["audio_duration"],
            audio_normalize=data_conf["dataset"]["audio_normalize"],
            audio_overlap=data_conf["dataset"]["audio_overlap"],
            audio_include_ratio=data_conf["dataset"]["audio_include_ratio"],
            audio_context_duration=data_conf["dataset"]["audio_context_duration"],
            ipu_min_time=data_conf["dataset"]["ipu_min_time"],
            ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
            sample_rate=data_conf["dataset"]["sample_rate"],
            vad_hop_time=data_conf["dataset"]["vad_hop_time"],
            vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
            vad_history=data_conf["dataset"]["vad_history"],
            vad_history_times=data_conf["dataset"]["vad_history_times"],
            shuffle_training_data=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        dm.prepare_data()
        return dm

    @torch.no_grad()
    def extract_sample_features_for_kmeans(self):
        """Extracts features (to RAM) with possibility to save to disk"""
        # always extract features from the training set
        if self.encoder is None:
            self.encoder = self._load_model()

        pbar = tqdm(
            total=int(self.args.kmeans_max_feature_memory * 1000),
            desc="Kmean train data",
            leave=False,
        )

        i = 0
        t = time.time()
        cum_memory = 0
        all_feats = []
        isnan = 0
        for batch in self.dm.train_dataloader():
            feats = self.encoder.encode(batch["waveform"].to(self.encoder.device))
            if feats.isnan().sum() > 0:
                isnan += 1
                continue
            feats = einops.rearrange(feats, "b t d -> (b t) d")
            m = get_memory(feats, "gb")
            pbar.update(int(m * 1000))
            cum_memory += m
            all_feats.append(feats.cpu())  # store as point cloud
            i += 1
            if cum_memory >= self.args.kmeans_max_feature_memory:
                break
        pbar.close()

        t = time.time() - t
        all_feats = torch.cat(all_feats)
        torch.save(all_feats, self.paths["k_training_features"])

        # Spinner
        m = round(get_memory(all_feats, "gb"), 2)
        # s = tuple(all_feats.shape)
        t = round(t, 2)
        spinner = Halo("")
        spinner.succeed(
            f"Features ({m}Gb / {t}s) -> {self.paths['k_training_features']}"
        )

        if isnan > 0:
            print("NaN batches: ", isnan)
        return all_feats

    def train_kmeans(self, features):
        if args.kmeans_algorithm == "faiss":
            raise NotImplementedError("FAISS Kmean is not implemented...")

        # if args.kmean_algorithm == "sklearn":

        kmean = KmeanSKLearn(
            k=self.model_config["quantizer"]["n_codes"],
            init=args.init,
            batch_size=args.k_batch_size,
            tol=args.tol,
            n_init=args.n_init,
            reassignment_ratio=args.reassignment_ratio,
            max_no_improvement=args.max_no_improvement,
            random_state=0,
            compute_labels=True,
            init_size=None,
        )

        spinner = Halo(text="K-means Training")
        spinner.start()
        kmean.train(features.cpu())
        inertia = round(kmean.inertia(features.cpu()), 3)
        kmeans_vectors = kmean.get_vectors()
        torch.save(kmeans_vectors, self.paths["k_vectors"])

        spinner.succeed(
            f"K-means Vectors (inertia={inertia}) -> {self.paths['k_vectors']}"
        )
        return kmeans_vectors

    @torch.no_grad()
    def save_dataloader_samples(
        self, quantizer, dataloader, root, with_audio_root=None, n_audio=20
    ):
        # Load model if necesary
        if self.encoder is None:
            self.encoder = self._load_model()

        i, n_batch, n_audio_done = 0, 0, 0
        isnan = 0
        spinner = Halo(text=f"Unit Dataset {basename(root)}", spinner="dots")
        spinner.start()
        for batch in dataloader:

            feats = self.encoder.encode(batch["waveform"].to(self.encoder.device))
            if feats.isnan().sum() > 0:
                isnan += 1
                continue

            _, qidx = quantizer(feats)

            batch = self.encoder.fix_batch_hz(batch)
            do_audio = True  # flag to take audio_samples from different batch
            # save each dialog segment
            for j, q_tmp in enumerate(qidx):
                # make sure they are of equal size
                sample = {
                    "q": q_tmp.cpu(),
                    "vad_label": batch["vad_label"][j, : len(q_tmp)].cpu(),
                    "vad": batch["vad"][j, : len(q_tmp)],
                    "vad_history": batch["vad_history"][j, : len(q_tmp)],
                }
                torch.save(sample, join(root, f"k_{quantizer.k}_{i}.pt"))

                # Save subset for video/training/visualization
                if with_audio_root is not None and do_audio:
                    if n_audio_done < n_audio:
                        sample["waveform"] = batch["waveform"][j]
                        torch.save(
                            sample, join(with_audio_root, f"k_{quantizer.k}_{i}.pt")
                        )
                        n_audio_done += 1
                        do_audio = False
                i += 1

            n_batch += 1
            if (
                self.args.samples_max_batches > 0
                and self.args.samples_max_batches == n_batch
            ):
                break

        spinner.succeed(f"Unit Dataset {basename(root)}")

    def extract_unit_dataset(self, kmean_vectors):
        """

        Extracts all samples in the training dataloader
        Name: extract_kmeans_feature_dataset


        Args:
            model:              nn.Module, Encoder model
            dm:                 LightningDataModule
            kmean_vectors:      torch.Tensor, kmeans codebook
            args:               Namespace, parsed ArgumentParser
        """

        # Init K-means embedding/codebook
        k, d = kmean_vectors.size()
        quantizer = KMeanEmbedding(k=k, dim=d, vectors=kmean_vectors)

        if torch.cuda.is_available():
            quantizer = quantizer.to("cuda")

        # Train
        self.save_dataloader_samples(
            quantizer, self.dm.train_dataloader(), self.paths["train_files_root"]
        )

        # Validation + subset (for visualization during training)
        self.save_dataloader_samples(
            quantizer,
            self.dm.val_dataloader(),
            self.paths["val_files_root"],
            with_audio_root=self.paths["val_subset"],
            n_audio=20,
        )

    def kmeans_vectors(self):
        print("extract kmeans")
        if self.args.overwrite_kmeans:
            features = self.extract_sample_features_for_kmeans()
            kmeans_vectors = self.train_kmeans(features)
        else:
            if exists(self.paths["k_vectors"]):
                # Load Kmeans_vectors
                kmeans_vectors = torch.load(self.paths["k_vectors"])
                Halo(text="").succeed(
                    f"K-means Vectors Loaded -> {self.paths['k_vectors']}"
                )
            else:
                if exists(self.paths["k_training_features"]):
                    features = torch.load(self.paths["k_training_features"])
                    Halo("").succeed(
                        f"Loaded Features -> {self.paths['k_training_features']}"
                    )
                else:
                    features = self.extract_sample_features_for_kmeans()
                kmeans_vectors = self.train_kmeans(features)
        return kmeans_vectors

    def build(self):
        self.start_time = time.time()
        self.dm.setup("fit")
        kmeans_vectors = self.kmeans_vectors()

        # 3. Extract kmeans-indices (and labels) over entire train-dataset
        if self.args.extract_dataset:
            self.extract_unit_dataset(kmeans_vectors)

        self.save_configs()
        self.end_time = time.time()
        duration = round(self.end_time - self.start_time, 2)
        print("====================================")
        print(f"Finished in {duration} seconds")
        print("====================================")

    @staticmethod
    def get_args():
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = ArgumentParser()
        parser.add_argument("--savepath", default="tmp_data", type=str)
        parser.add_argument(
            "--conf", default="conv_ssl/config/encoder_hubert.yaml", type=str
        )
        parser.add_argument("--k_vectors_path", default=None, type=str)
        parser.add_argument("--k_training_features_path", default=None, type=str)
        parser.add_argument(
            "--layer", default=6, type=int, help="gslm used 6th layer for hubert_base"
        )
        parser.add_argument("--extract_dataset", action="store_true")
        parser.add_argument("--overwrite_kmeans", action="store_true")
        parser.add_argument("--overwrite_features", action="store_true")
        parser.add_argument(
            "--samples_max_batches",
            default=-1,
            type=int,
            help="Maximum batches to extract (used mainly for debug)",
        )
        parser.add_argument(
            "--kmeans_algorithm",
            default="sklearn",
            type=str,
            help="Which algorithm to use for Kmeans training",
        )
        parser.add_argument(
            "--kmeans_max_feature_memory",
            default=20,
            type=float,
            help="Maximum memory (in Gb) to extract for kmeans training",
        )
        parser.add_argument(
            "--kmeans_max_batches",
            default=-1,
            type=int,
            help="Maximum batches to extract. No max if negative values",
        )

        tmp_args, _ = parser.parse_known_args()
        if tmp_args.kmeans_algorithm == "faiss":
            raise NotImplementedError("FAISS not implemented")
        else:
            from conv_ssl.models.kmean import KmeanSKLearn

            parser = KmeanSKLearn.add_model_specific_args(parser)

        parser = DialogAudioDM.add_data_specific_args(parser)
        return parser.parse_args()


if __name__ == "__main__":
    args = SegmentDatasetBuilder.get_args()
    builder = SegmentDatasetBuilder(args)
    builder.build()
