from utils.dominant_color import extractDominantColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

def detect_color(colored_mask, reference_red_lab, reference_blue_lab):
    # r_xywh = scale_coords(img.shape[2:], box.xywh.clone(), im0.shape, kpt_label=False).round()
    
    # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
    dominantColors = extractDominantColor(colored_mask,hasThresholding=True)

    # #Show in the dominant color information
    # print("Color Information")
    # prety_print_data(dominantColors)

    # #Show in the dominant color as bar
    # print("Color Bar")
    # colour_bar = plotColorBar(dominantColors)
    # plt.axis("off")
    # plt.imshow(colour_bar)
    # plt.show()

    red_color_diff = []
    blue_color_diff = []
    for dc in dominantColors:
        color = tuple(map(int,(dc['color'])))
        color = [round(float(i)/255.0, 2) for i in color]
        color_srgb = sRGBColor(color[0], color[1], color[2])
        color_lab = convert_color(color_srgb, LabColor)

        # Find the color difference
        red_color_diff.append(delta_e_cie2000(reference_red_lab, color_lab))
        blue_color_diff.append(delta_e_cie2000(reference_blue_lab, color_lab))

    delta_red = min(red_color_diff)
    delta_blue = min(blue_color_diff)

    player_id = None
    if delta_red < delta_blue:
        player_id = 1
    elif delta_blue < delta_red:
        player_id = 2

    # pose_db.insert_into_pose_table(
    #     frame_num=frame,
    #     player_num= player_id,
    #     bbox_coords= box.xyxy.tolist(),
    #     feet_coords= [int(x_center_below), int(y_center_below)],
    #     position=position
    # )

    # if(player_id not in seg_logs['segmentation'][frame]):
    #     seg_logs['segmentation'][frame][player_id] = []

    # # Add Tracked Person to Logs
    # player_entry = {
    #     "player_id": str(player_id),
    #     "bbox_coords": box.xyxy.tolist(),
    #     "feet_coords": [int(x_center_below), int(y_center_below)],
    #     "position": position,
    #     'action': None,
    #     'player_with_basketball': None
    # }

    # seg_logs['segmentation'][frame][player_id].append(player_entry)

    # label = "player: " + str(player_id)
    # old_label = names[int(box.cls)]

    # plot_one_box(r_xyxy.cpu().numpy()[0], im0, colors[int(box.cls)], f'{label} {float(box.conf):.3}')
