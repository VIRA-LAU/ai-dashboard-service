from moviepy.editor import *

from shared.helper.json_helpers import parse_json

def add_score(video, score_asset, scores):
    x = 877
    y = 800
    score_composites=[]
    frames = scores['frames']
    for frame in frames:
        # print(frames[frame]["team1"]["score"])
        team1_score_text = TextClip(str(frames[frame]['team1']['score']),font="Space Grotesk Bold", fontsize=30, color='white')
        team2_score_text = TextClip(str(frames[frame]['team2']['score']),font="Space Grotesk Bold", fontsize=30, color='white')

        # score_asset = score_asset.set_position(x,y)
        team1_score_text = team1_score_text.set_pos((x-100,y))
        team2_score_text = team2_score_text.set_pos((x+100,y))

        score_clip_composite = CompositeVideoClip([score_asset, team1_score_text, team2_score_text])
        score_clip_composite = score_clip_composite.set_start(int(frame)/30)
        score_composites.append(score_clip_composite)

    score_clip = CompositeVideoClip(score_composites)

    video_scores = CompositeVideoClip([video, score_clip])
    video_scores.duration = video.duration
    video_scores.write_videofile("datasets/post_process/exported/test.mp4", fps=30,codec='libx264')

    return video_scores


def add_lineups(video, lineups, team):
    x=500
    y=700
    text_composites=[]
    for player in team["players"]:
        number_txt = TextClip(player["number"],font="Space Grotesk", fontsize=30, color='black')
        first_name_txt = TextClip(player["first name"],font="Space Grotesk", fontsize=30, color='black')
        last_name_txt = TextClip(player["last name"],font="Space Grotesk", fontsize=30, color='black')

        number_txt = number_txt.set_pos(x,y)
        first_name_txt = first_name_txt.set_pos(x+70+number_txt.w,y)
        last_name_txt = last_name_txt.set_pos(x+70+last_name_txt.w,y)

        text_composites.append(number_txt, first_name_txt, last_name_txt)

    text_clip = CompositeVideoClip(text_composites)

    video_lineup = CompositeVideoClip([video, # starts at t=0
                            lineups.set_start(2).crossfadein(1),
                            text_clip.set_start(2).crossfadein(1.5)])
    
    video_lineup = video_lineup.set_duration(10).crossfadeout(2)
    
    return video_lineup


def load_clips(video_dir, lineups_asset, stats_asset, score_asset):
    video = VideoFileClip(video_dir)
    # lineups = VideoFileClip(lineups_asset)
    lineups = ''
    stats = ''
    # stats = VideoFileClip(stats_asset)
    score = ImageClip(score_asset)

    return video, lineups, stats, score


if __name__ == "__main__":
    video_dir = 'datasets/post_process/vids/DSC_0007.MOV'
    lineups_asset_dir = 'assets/templates/' 
    stats_asset_dir = 'assets/templates/'
    score_asset_dir = 'assets/templates/Scoreboard.png'
    video, lineups_asset, stats_asset, score_asset = load_clips(video_dir, lineups_asset_dir, stats_asset_dir, score_asset_dir)
    scores = parse_json("assets/post_process.json")["points_scored"]
    video_scores = add_score(video, score_asset, scores)