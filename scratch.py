''' Convert image to HSV'''
hsv_image = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2HSV)

'''Color Masks'''
# green_lower = np.array([50, 100, 100])
# green_upper = np.array([70, 255, 255])

# white_lower = np.array([0, 0, 200])
# white_upper = np.array([180, 50, 255])

red_lower = np.array([160,20,70])
red_upper = np.array([190, 255, 255])

blue_lower = np.array([101,50,38])
blue_upper = np.array([110, 255, 255])

# lower range of red color in HSV
red_lower = np.array([0, 50, 50], np.uint8) 
red_upper = np.array([150, 255, 255], np.uint8) 
red_mask = cv2.inRange(player_roi, red_lower, red_upper) 

kernel = np.ones((5, 5), "uint8") 

red_mask = cv2.dilate(red_mask, kernel) 
res_red = cv2.bitwise_and(player_roi, player_roi,  
                        mask = red_mask) 

contours, hierarchy = cv2.findContours(red_mask, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE) 

if(len(contours)):
    print("RED")

# for pic, contour in enumerate(contours): 
#     area = cv2.contourArea(contour) 
#     if(area > 300): 
#         x, y, w, h = cv2.boundingRect(contour) 
#         player_roi = cv2.rectangle(player_roi, (x, y),  
#                                 (x + w, y + h),  
#                                 (0, 0, 255), 2) 
        
#         cv2.putText(player_roi, "Red Colour", (x, y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
#                     (0, 0, 255))  
#         print("RED")
        
# cv2.imshow("Multiple Color Detection in Real-TIme", player_roi) 

if (player_roi>red_lower).all() and (player_roi<red_upper).all():
    print("RED")

# Get RGB data from image 
# hist = cv2.calcHist([player_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) 
# blue_color = cv2.calcHist([player_roi], [0], None, [256], [0, 256]) 
# red_color = cv2.calcHist([player_roi], [2], None, [256], [0, 256]) 

# blue_threshold = 35000
# red_threshold = 35000

# blue_bin_counts = blue_color.sum()
# red_bin_counts = red_color.sum()

# img = im.fromarray(player_roi)
# dominantcolor = DominantColor(img)
# print(dominantcolor.dominant_color)
# print(dominantcolor.rgb)

player_id = None
# if(dominantcolor.dominant_color=='r'):
#     player_id = 1
#     print('player 1')
# elif(dominantcolor.dominant_color=='b'):
#     player_id = 2
#     print('player 2')



                    player_roi = im.fromarray((colored_mask * 255).astype(np.uint8))
                    dominantcolor = DominantColor(player_roi)
                    print(dominantcolor.dominant_color)
                    print(dominantcolor.rgb)







# mask = r.masks[idx].cpu().data.numpy().transpose(1, 2, 0)

# # Get the size of the original image (height, width, channels)
# h2, w2, c2 = r.orig_img.shape

# # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
# mask = cv2.resize(mask, (w2, h2))

red_hist = cv2.calcHist([player_roi], [2], None, [256], [0, 256])
blue_hist = cv2.calcHist([player_roi], [0], None, [256], [0, 256])

count_red = red_hist.sum()
count_blue = blue_hist.sum()

print(count_red)
print(count_blue)

# # Create masks to isolate red and blue colors
# red_mask = cv2.inRange(player_roi, red_lower, red_upper)
# blue_mask = cv2.inRange(player_roi, blue_lower, blue_upper)

# red_pixels = cv2.countNonZero(red_mask)
# blue_pixels = cv2.countNonZero(blue_mask)

# # Compare the number of red and blue pixels to determine dominance
# if red_pixels > blue_pixels:
#     print("Red is more dominant.")
# # else:
# #    print("Blue is more dominant.")

# masked = cv2.bitwise_and(player_roi, player_roi, mask=mask)


def sef():
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


    results = yolo.track(
        source=source,
        stream=stream,
        persist=False,

        conf=conf,
        iou=iou,
        show=show,
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


                    if save_id_crops:
                        save_one_box(
                            d.xyxy,
                            det.orig_img.copy(),
                            file=(
                                yolo.predictor.save_dir / 'crops' /
                                str(int(d.cls.cpu().numpy().item())) /
                                str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                            ),
                            BGR=True
                        )



                    '''Color Histogram'''
                    # Get RGB data from image 
                    blue_color = cv2.calcHist([colored_mask], [0], None, [256], [0, 256]) 
                    red_color = cv2.calcHist([colored_mask], [1], None, [256], [0, 256]) 
                    green_color = cv2.calcHist([colored_mask], [2], None, [256], [0, 256]) 

                    # # combined histogram 
                    # plt.title("Histogram of all RGB Colors") 
                    # plt.hist(blue_color, color="blue") 
                    # plt.hist(green_color, color="green") 
                    # plt.hist(red_color, color="red") 
                    # plt.show() 


def seg():
    # for frame_idx, r in enumerate(results):

    #     if r.boxes.data.shape[1] == 7:
    #         if yolo.predictor.source_type.webcam or source.endswith(VID_FORMATS):
    #             p = yolo.predictor.save_dir / 'mot' / (source + '.txt')
    #             yolo.predictor.mot_txt_path = p
    #         elif 'MOT16' or 'MOT17' or 'MOT20' in source:
    #             p = yolo.predictor.save_dir / 'mot' / (Path(source).parent.name + '.txt')
    #             yolo.predictor.mot_txt_path = p

    #         if save_mot:
    #             write_mot_results(yolo.predictor.mot_txt_path, r, frame_idx)
            
    #         for box_idx, d in enumerate(r.boxes):
    #             if r.masks is not None:
    #                 masks = r.masks.data.cpu()
    #                 for seg, box in zip(masks.data.cpu().numpy(), boxes):

    #                     seg = cv2.resize(seg, (w, h))
    #                     img = overlay(img, seg, colors[int(box.cls)], 0.4)
                        
    #                     xmin = int(box.data[0][0])
    #                     ymin = int(box.data[0][1])
    #                     xmax = int(box.data[0][2])
    #                     ymax = int(box.data[0][3])
    #                 mask_raw = r.masks[box_idx].cpu().data.numpy().transpose(1, 2, 0)   # masks, (N, H, W)
    #                 mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

    #                 h2, w2, c2 = r.orig_img.shape
    #                 mask = cv2.resize(mask_3channel, (w2, h2))

    #                 # Convert BGR to HSV
    #                 hsv_img = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2HSV)
    #                 hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    #                 # Define range of brightness in HSV
    #                 lower_black = np.array([0,0,0])
    #                 upper_black = np.array([0,0,1])

    #                 # Create a mask. Threshold the HSV image to get everything black
    #                 threshold = cv2.inRange(hsv_mask, lower_black, upper_black)

    #                 # Invert the mask to get everything but black
    #                 threshold = cv2.bitwise_not(threshold)

    #                 # Apply the mask to the original image
    #                 segmented_img = cv2.bitwise_and(hsv_img, hsv_img, mask=threshold)

    #                 cv2.imshow("Segmented Image", segmented_img)
    #                 cv2.waitKey(0)
    #                 cv2.destroyAllWindows()

    #             if save_id_crops:
    #                 save_one_box(
    #                     d.xyxy,
    #                     r.orig_img.copy(),
    #                     file=(
    #                         yolo.predictor.save_dir / 'crops' /
    #                         str(int(d.cls.cpu().numpy().item())) /
    #                         str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
    #                     ),
    #                     BGR=True
    #                 )

    #         '''for idx, det in enumerate(r.boxes):
    #             mask = r.masks[idx].cpu().data.numpy().transpose(1, 2, 0)
    #             x, y, w, h = det.xywh.cpu().numpy()[0]

    #             x_center_below = x + w/2
    #             y_center_below = y + h

    #             player_roi = r.orig_img[int(y):int(y+h/2), int(x):int(x+w), :]

    #             player_roi = cv2.bitwise_and(player_roi, player_roi, mask=mask)
    #             cv2.imshow(player_roi)
                
    #             player_id = None

    #             yUp = polyUp(x_center_below)
    #             yDown = polyDown(x_center_below)

    #             position=""
    #             if yUp > y_center_below and yDown < y_center_below:
    #                 position = "2_points"
    #             else:
    #                 position = "3_points"

    #             pose_db.insert_into_pose_table(
    #                 frame_num=frame_idx,
    #                 player_num= player_id,
    #                 bbox_coords= d.xyxy.tolist(),
    #                 feet_coords= [x_center_below, y_center_below],
    #                 position=position
    #             )'''

    # if save_mot:
    #     print(f'MOT results saved to {yolo.predictor.mot_txt_path}')