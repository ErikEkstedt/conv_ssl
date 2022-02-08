from vad_turn_taking import DialogEvents, VAD, VadLabel


if __name__ == "__main__":

    from datasets_turntaking import DialogAudioDM

    FRAME_HZ = 100
    data_conf = DialogAudioDM.load_config()
    # data_conf["dataset"]["type"] = "sliding"
    DialogAudioDM.print_dm(data_conf)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        vad_hz=FRAME_HZ,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=32,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.val_dataloader()))

    vad = batch["vad"]  # vad: (B, N, 2)
    print("vad: ", tuple(vad.shape))

    valid = DialogEvents.on_silence(
        vad, start_pad=5, target_frames=10, horizon=100, min_context=50
    )
