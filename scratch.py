
def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = os.path.join('yolo_tracking/boxmot/configs',(predictor.custom_args.tracking_method + '.yaml'))
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

@torch.no_grad()
def instance_segmentation(weights: str = 'yolov8.pt',
                        reid_model: Path = '',
                        tracking_method: str = 'deepocsort',
                        source: str='',
                        conf: float=0.6,
                        iou: float=0.45,
                        show: bool=True,
                        img_size: int=640,
                        stream: bool=False,
                        device= torch.device("cuda:0"),
                        half: bool=True,
                        show_conf: bool=False,
                        save_txt: bool=False,
                        show_labels: bool=True,
                        save: bool=True,
                        save_mot: bool=False,
                        save_id_crops: bool=True,
                        verbose: bool=True,
                        exist_ok: bool=False,
                        save_dir: str = 'datasets/videos_inferred',
                        name: str = 'segmentation',
                        classes: int = 0,
                        per_class: bool = False,
                        vid_stride: int = 1,
                        line_width: int = 3):
    
    args = parse_segmentation(weights=weights,
                        reid_model=reid_model,
                        tracking_method=tracking_method,
                        source=source,
                        conf=conf,
                        iou=iou,
                        show=show,
                        img_size=img_size,
                        device=device,
                        half=half,
                        show_conf=show_conf,
                        save_txt=save_txt,
                        show_labels=show_labels,
                        save=save,
                        save_mot=save_mot,
                        save_id_crops=save_id_crops,
                        verbose=verbose,
                        exist_ok=exist_ok,
                        save_dir=save_dir,
                        name=name,
                        classes=classes,
                        per_class=per_class,
                        vid_stride=vid_stride,
                        line_width=line_width)

    yolo = YOLO(weights)

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=stream,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    # tracking_method=tracking_method
    # reid_model=reid_model
    # device=device
    # half=True
    # per_class=False
    # persist=True
    # data = (tracking_method, reid_model, device, half, per_class, persist)

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    yolo.predictor.custom_args = args

    for frame_idx, r in enumerate(results):
        if r.boxes.data.shape[1] == 7:
            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')



def instance_segmentation():
    model = YOLO('yolov8n-seg.pt')

    video_path = 'datasets/videos_input/PhoneDatasetOne_1.mkv'
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, device=0)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()



def on_predict_start(predictor, data: tuple = None):
    tracking_method, reid_model, device, half, per_class, persist = data

    assert tracking_method in TRACKERS, \
        f"'{tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = os.path.join('yolo_tracking/boxmot/configs',(tracking_method + '.yaml'))
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            tracking_method,
            tracking_config,
            reid_model,
            device,
            half,
            per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers





def on_predict_start(predictor, data=tuple):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    tracking_method, reid_model, device, half, per_class, persist = data

    assert tracking_method in TRACKERS, \
        f"'{tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = os.path.join('yolo_tracking/boxmot/configs',(tracking_method + '.yaml'))
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            tracking_method,
            tracking_config,
            reid_model,
            device,
            half,
            per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


def instance_segmentation(weights: str = 'yolov8.pt',
                        reid_model: Path = '',
                        tracking_method: str = 'deepocsort',
                        source: str='',
                        conf: float=0.6,
                        iou: float=0.45,
                        show: bool=True,
                        img_size: int=640,
                        stream: bool=False,
                        device= torch.device("cuda:0"),
                        half: bool=True,
                        show_conf: bool=False,
                        save_txt: bool=False,
                        show_labels: bool=True,
                        save: bool=True,
                        save_mot: bool=False,
                        save_id_crops: bool=True,
                        verbose: bool=True,
                        exist_ok: bool=False,
                        save_dir: str = 'datasets/videos_inferred',
                        name: str = 'segmentation',
                        classes: int = 0,
                        per_class: bool = False,
                        vid_stride: int = 1,
                        line_width: int = 3):
    
    args = parse_segmentation(weights=weights,
                        reid_model=reid_model,
                        tracking_method=tracking_method,
                        source=source,
                        conf=conf,
                        iou=iou,
                        show=show,
                        img_size=img_size,
                        device=device,
                        half=half,
                        show_conf=show_conf,
                        save_txt=save_txt,
                        show_labels=show_labels,
                        save=save,
                        save_mot=save_mot,
                        save_id_crops=save_id_crops,
                        verbose=verbose,
                        exist_ok=exist_ok,
                        save_dir=save_dir,
                        name=name,
                        classes=classes,
                        per_class=per_class,
                        vid_stride=vid_stride,
                        line_width=line_width)

    yolo = YOLO(weights)

    results = yolo.track(
        source=source,
        conf=conf,
        iou=iou,
        show=show,
        stream=stream,
        device=device,
        show_conf=show_conf,
        save_txt=save_txt,
        show_labels=show_labels,
        save=save,
        verbose=verbose,
        exist_ok=exist_ok,
        project=save_dir,
        name=name,
        classes=classes,
        imgsz=img_size,
        vid_stride=vid_stride,
        line_width=line_width
    )

    tracking_method=tracking_method
    reid_model=reid_model
    device=device
    half=True
    per_class=False
    persist=True
    data = (tracking_method, reid_model, device, half, per_class, persist)

    yolo.add_callback('on_predict_start', partial(on_predict_start, data=data))
    
    yolo.predictor.custom_args = args

    for frame_idx, r in enumerate(results):
        if r.boxes.data.shape[1] == 7:
            if source.endswith(VID_FORMATS):
                p = save_dir / 'mot' / (source + '.txt')
                mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in source:
                p = save_dir / 'mot' / (Path(source).parent.name + '.txt')
                mot_txt_path = p

            if save_mot:
                write_mot_results(
                    mot_txt_path,
                    r,
                    frame_idx,
                )

            if save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    if save_mot:
        print(f'MOT results saved to {mot_txt_path}')
