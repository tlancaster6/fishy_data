import subprocess, os, cv2, argparse, sys, logging
from datetime import datetime


def file_namer(projectID, v_file, step, frame_count, fps):
    """
    -creates a filename in the format <parent-projectID>_<parent-video-name>_<step(min)>_<frame>_<hr-min-sec.ms>.jpg
    :return: {str} <filename>
    """
    time = frame_count/fps
    hrs = int(time // 3600)
    mins = int(time // 60 % 60)
    secs = round(time % 60, 2)
    time = '{:02d}-{:02d}-{:05.2f}'.format(hrs, mins, secs)
    filename = '{}_{}_{}_{}_{}.jpg'.format(projectID, v_file.split('.')[0], step, frame_count, time)
    return filename


def get_fish_data(projectIDs, videoIDs, step, drive):
    """
    -creates image dataset in dropbox 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/__TrainingData/CichlidDetection/
    Training2021/Images_<step> with log file for each set
    -if image file exists, prints location
    """
    # if no step was specified (step = 0), exit
    if step == 0:
        sys.exit('Error... Please define a step > 0')
    # initialize
    file_dup = False
    species = 'Unknown'
    species_d = {'rock': ['rock', 'mz', 'rs', 'kl'], 'sand': ['sand', 'ti', 'mc', 'cv'], 'rs': ['rs'], 'cv': ['cv'], 'ti': ['ti'],
               'mc': ['mc'], 'mz': ['mz'], 'kl': ['kl'], 'bh': ['bh']}
    # {'1 ec?': ['ec'], '6 mcst?': ['mcst'], '2 bs?': ['bs']}
    f_dir = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/'
    dataset_dir = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/__TrainingData/CichlidDetection/Training2021/Images_{}'.format(step)
    temp_dir = drive+'/temp'
    subprocess.run('rclone mkdir {}/Rock/'.format(dataset_dir), shell=True)
    subprocess.run('rclone mkdir {}/Sand/'.format(dataset_dir), shell=True)
    subprocess.run('rclone mkdir {}/Logs/'.format(dataset_dir), shell=True)
    os.makedirs(temp_dir, mode=0o777, exist_ok=True)
    logging.basicConfig(filename='{}/gfd_log.txt'.format(temp_dir), level=logging.DEBUG)
    logger = logging.getLogger()
    exist_log = subprocess.run('rclone lsf --include "*.txt" {}/'.format(dataset_dir), shell=True,
                               capture_output=True, encoding='utf-8').stdout
    # if log does not exist, initialize log file in dropbox
    if not exist_log:
        with open('{}/Images_{}_log.txt'.format(temp_dir, step), 'w') as log_file:
            print('File, Step, Frame, Time, Date', file=log_file)
        subprocess.run('rclone move {}/Images_{}_log.txt {}/'.format(temp_dir, step, dataset_dir), shell=True)
    # initialize projects, a list of all projectIDs to use
    projects = []
    for project in projectIDs:
        # if no projects, exit
        if not project:
            # not sure if this is the best way to raise this case
            sys.exit('Error... Please enter a valid project ID.')
        # initialize projectIDs if -p = 'all'
        elif project.lower() == 'all':
            # add --
            projects = subprocess.run('rclone lsf --dirs-only {}'.format(f_dir), shell=True, capture_output=True,
                                      encoding='utf-8').stdout.split('\n')
            # del empty str
            del projects[-1]
            # remove trailing '/'
            projects = [p[:-1] for p in projects]
        elif any(project.lower() in s for s in species_d):
            # use s ok or ss?
            print('here')
            print(','.join(ss for ss in species_d[project.lower()]))
            print('rclone lsf {0} --ignore-case --include "{{{1}}}**"'.format(
                f_dir, ','.join(ss for ss in species_d[project.lower()])))
            projects = subprocess.run('rclone lsf {0} --ignore-case --include "{{{1}}}**"'.format(
                f_dir, ','.join(ss for ss in species_d[project.lower()])),
                shell=True, capture_output=True, encoding='utf-8').stdout.split('\n')
            print('there')
            del projects[-1]
            projects = [p[:-1] for p in projects]
            print(projects)
        else:
            projects.append(project)
    # for all specified videos in desired projects, add frame images to dataset and log new images
    print(projects)
    for project in projects:
        # print project
        print(project)
        # is project rock or sand
        for i in species_d['rock']:
            if i in project.lower():
                species = 'Rock'
        for i in species_d['sand']:
            if i in project.lower():
                species = 'Sand'
        # if all, list all .mp4 filenames in project and store in videos
        if videoIDs.lower() == 'all':
            videos = subprocess.run(
                'rclone lsf --include "*.mp4" {}{}/Videos/'.format(f_dir, project),
                shell=True, capture_output=True, encoding='utf-8').stdout.split('\n')
            # del empty str
            del videos[-1]
        else:
            # if not all, initialize videos accordingly
            videos = []
            for v in videoIDs:
                if ':' in v:
                    v = list(range(int(v.split(':')[0]), int(v.split(':')[1])+1))
                    while v:
                        videos.append('{:04d}_vid.mp4'.format(v.pop(0)))
                else:
                    videos.append('{:04d}_vid.mp4'.format(int(v)))
        # for .mp4 in videos, if exists create temp directory to write in, download .mp4 to temp,
        # create cv2.VideoCapture object, store fps, create image np.array, write np.array to .jpg in temp,
        for video in videos:
            if not video:
                print('Warning... No .mp4 files found in {}.'.format(project))
                logger.info('No .mp4 files found in {}.'.format(project))
            else:
                os.makedirs(temp_dir, mode=0o777, exist_ok=True)
                subprocess.run('rclone copy -P {}{}/Videos/{} {}'.format(f_dir, project, video, temp_dir),
                               shell=True)
                f_vid = cv2.VideoCapture('{}/{}'.format(temp_dir, video))
                # fps sometimes not whole number
                fps = round(f_vid.get(cv2.CAP_PROP_FPS))
                success, frame = f_vid.read()
                count = 0
                while success:
                    try:
                        # write frame image to temp,
                        # filename format <parent-video-name>_<step(min)>_<frame>_<hr-min-sec.ms>.jpg
                        result = cv2.imwrite('{}/{}'.format(temp_dir, file_namer(project, video, step, count, fps)), frame)
                        # if successful, copy .jpg to dropbox, copy log from dropbox to temp, write to log in temp,
                        # copy log to dropbox, remove .jpg and log
                        if result:
                            # if frame_img already exists in dataset, do not copy image, do not write log
                            if file_namer(project, video, step, count, fps) in subprocess.run(
                                    'rclone lsf --include "*.jpg" {}'.format(dataset_dir), shell=True,
                                    capture_output=True, encoding='utf-8').stdout.split('\n')[:-1]:
                                file_dup = True
                                logger.info('{} was not saved because it already exists in .../Images_{}/.'.format(
                                    file_namer(project, video, step, count, fps), step))
                                os.remove('{}/{}'.format(temp_dir, file_namer(project, video, step, count, fps)))
                            else:
                                # move image from temp to dropbox
                                subprocess.run(
                                    'rclone move {}/{} {}/{}'.format(
                                        temp_dir, file_namer(project, video, step, count, fps), dataset_dir, species),
                                    shell=True
                                )
                                # move log.txt from dropbox to temp
                                subprocess.run(
                                    'rclone move {}/Images_{}_log.txt {}'.format(
                                        dataset_dir, step, temp_dir),
                                    shell=True
                                )
                                # write log file to temp
                                with open('{}/Images_{}_log.txt'.format(temp_dir, step), 'a') as log_file:
                                    # format '<file>, <parent-project>, <step>, <frame>, <time>, <date>\n'
                                    print('{}, {}, {}, {}, {}'.format(
                                        file_namer(project, video, step, count, fps),
                                        step,
                                        count,
                                        file_namer(
                                            project, video, step, count, fps).split('_')[-1].split('.jpg')[0],
                                        datetime.now()
                                                                  ), file=log_file)
                                # copy log.txt from temp to dropbox
                                subprocess.run('rclone move {}/Images_{}_log.txt {}'.format(
                                    temp_dir, step, dataset_dir), shell=True)
                        else:
                            print('Unable to write {}'.format(file_namer(project, video, step, count, fps)))
                        count += fps*60*step
                        f_vid.set(cv2.CAP_PROP_POS_FRAMES, count)
                        success, frame = f_vid.read()
                    except Exception as e:
                        logger.error(e)
                f_vid.release()
                # remove .mp4 from temp
                os.remove('{}/{}'.format(temp_dir, video))
        if file_dup:
            print('Info... Some images in {} already exist. Check gfd_log for info.'.format(project))
    subprocess.run('rclone lsf {}'.format(temp_dir), shell=True)
    subprocess.run('rclone move {}/gfd_log.txt {}/Logs/ --create-empty-src-dirs'.format(temp_dir, dataset_dir), shell=True)
    # remove temp dir
    if os.path.exists(temp_dir):
        for temp_file in os.listdir(temp_dir):
            temp_path = os.path.join(temp_dir, temp_file)
            try:
                os.remove(temp_path)
            except OSError as e:
                print('Warning... Failed to remove temp dir.')
                logger.error(e)
    os.rmdir(temp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script creates unannotated datasets from fish videos in cichlidV'
                                                 'ideo:.../__ProjectData. Enter projectIDs, videoIDs, and a time interv'
                                                 'al to capture frames by.')
    parser.add_argument('-p', '--projectIDs', type=str, nargs='+', default='',
                        help='Enter desired project ID(s) separated by spaces. Alternatively, enter species name or abr'
                             'eveation separated by spaces or "all" , ex. "_newtray_CV1", "", "MC CV TI"')
    parser.add_argument('-v', '--videoIDs', type=str, nargs='+', default='all',
                        help='Enter desired video ID(s) and/or range(s) of IDs in format "<start>:<stop>" separated by '
                             'spaces, ex. "1" "1 2 5" "1:10" "1 3:5 7 12:15". VideoID correlates to XXXX_vid.mp4. If '
                             'none specified or multiple project IDs, defaults to all.')
    parser.add_argument('-s', '--step', type=int, default=0,
                        help='Enter the time interval in minutes (an integer) to clip frames from each video, ex. "10",'
                             ' "15", "30"...')
    parser.add_argument('-d', '--drive', type=str, default=os.getcwd(),
                        help='Enter an alternate location for temp dir, where files are downloaded. Default is cwd.')
    args = parser.parse_args()
    get_fish_data(args.projectIDs, args.videoIDs, args.step, args.drive)

#use rclone modified date to narrow down all projects, double check logfile date
#cvat
