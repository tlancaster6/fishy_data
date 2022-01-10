import cv2
import subprocess as sp
from pathlib import Path, PurePath, PurePosixPath
from datetime import datetime as dt
import shutil
import os
import argparse
from tqdm import tqdm

class VideoParser:

    def __init__(self, timestep, subset, min_year, min_month):
        self.timestep, self.subset, self.min_year, self.min_month = timestep, subset, min_year, min_month
        self.fm = FileManager(timestep, subset)
        self.fm.mkdir_local(self.fm.data_dir)

    def execute(self):
        """
        execute all steps in the video parsing pipeline using the current set of input arguments (subset, timestep,
        etc)
        """
        pids = self.get_pids()    # pids is short for Project Ids
        print('parsing videos')
        for pid in tqdm(pids):
            self.parse_project(pid)
        self.upload_imgs()

    def get_pids(self):
        """
        Return as a list all of the project ID's (pids) on Dropbox that meet the subset, min_year, and min_month
        conditions.
        """
        print('determining valid projects')
        # get all pids, i.e., all directory names in __ProjectData on DropBox
        all_pids = self.fm.ls_cloud(self.fm.data_dir, dirs_only=True)

        # handle the subset argument, with attention to special cases like 'rock', 'sand', and 'all'
        if self.subset == 'rock':
            search_strings = ['rock', 'mz', 'rs', 'kl']
        elif self.subset == 'sand':
            search_strings = ['sand', 'ti', 'mc', 'cv']
        elif (type(self.subset) is list) and all(type(x) is str for x in self.subset):
            search_strings = self.subset
        elif type(self.subset) is str:
            search_strings = [self.subset]
        else:
            raise Exception(f'invalid subset argument. Expected str or list of str, got {type(self.subset)}')

        # loop through all_pids, and for each check whether the creation date is valid and the pid contains one of the
        # search strings. If so, append that pid to valid_pids
        valid_pids = []
        for pid in tqdm(all_pids):    # tqdm displays a progress bar when the script is run
            # first, check that that pid contains one of the search strings, or that the chosen subset was 'all'
            if (self.subset == ['all']) or (any(s.lower() in pid.lower() for s in search_strings)):
                # next, check that the year and month of creation are >= the desired minimums
                year, month = self.check_creation_date(pid)
                if (year >= self.min_month) and (month >= self.min_month):
                    # if a pid meets all conditions, strip any residual trailing/leading slashes and append it
                    valid_pids.append(pid.strip('/'))
        print(f'{len(valid_pids)} valid projects located')
        return valid_pids

    def check_creation_date(self, pid):
        """
        Check a project's creation date robustly by downloading and reading its logfile
        """
        # form the relative path (i.e., relative to the local or cloud root) where we should find a given project's
        # logfile
        rpath = self.fm.data_dir / pid / 'Logfile.txt'
        # check that the logfile exists at that location. When you provide a path to a file to rclone lsf (instead of
        # a path to a directory) it returns a list containing the filename (in this case, Logfile.txt) if the file
        # exists, or an empty list if it does not.
        if len(self.fm.ls_cloud(rpath)) > 0:
            # download the logfile
            self.fm.download(rpath)
            # form the local path (here denoted lpath)
            lpath = (self.fm.local_root / rpath)
            # read the logfile line by line
            with open(lpath) as f:
                for line in f:
                    # Look for a line starting with 'MasterRecordInitialStart:'
                    if line.startswith('MasterRecordInitialStart:'):
                        # isolate the poriton of the line containing date/time information
                        t = line.split(': ')[-1].strip()
                        # convert the date/time string to a proper datetime object
                        t = dt.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
                        # break the loop to terminate reading of the file
                        break
            # delete the logfile
            self.fm.local_delete(rpath)
            # return the year and month of creation
            return t.year, t.month
        # if the logfile was not found, return 0 for year and month. This ensures that later checks of the form
        # month >= min_month and year >= min_year will return false
        else:
            return 0, 0

    def parse_project(self, pid):
        """
        parse all videos from a given project ID (pid) into images
        """
        # form the relative path to the project's video directory
        vid_dir = self.fm.data_dir / pid / 'Videos'
        # get the filenames for all of the .mp4 and .h264 videos in vid_dir
        vid_names = [n for n in self.fm.ls_cloud(vid_dir) if (('.mp4' in n) or ('.h264' in n))]
        for vname in vid_names:
            # form the relative path to the individual video file and download it from Dropbox
            vid_path = vid_dir / vname
            self.fm.download(vid_path)
            if '.h264' in vname:
                original = vname
                vname = original.replace('.h264', '.mp4')
                command = ['ffmpeg', '-r', '30', '-i', original, '-threads', '1', '-c:v', 'copy', '-r', '30', vname]
                sp.run(command)
                self.fm.local_delete(vid_path)
                vid_path = vid_dir / vname
            # generate a video capture object and extract the framerate
            cap = cv2.VideoCapture(str(self.fm.local_root / vid_path))
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            # convert the user-provided timestep (in minutes) to an equivalent frame-step
            frame_step = int(self.timestep * 60 * fps)
            count = 0
            # while there are still frames remaining in the video, save every nth frame as an image
            while cap.isOpened():
                # use the capture object to read a frame. ret will be True iff reading succeeded
                ret, frame = cap.read()
                if ret:
                    # generate a unique filename and relative path for the current frame
                    img_name = self.generate_filename(pid, vname, count, fps)
                    img_path = self.fm.local_root / self.fm.results_dir / img_name
                    # write the image to the local results directory
                    cv2.imwrite(str(img_path), frame)
                    # increment the count by frame_step, and force the video_capture object to advance its reading
                    # position to match count. This circumvents reading of the intermediate frames we will not be saving
                    count += frame_step
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                else:
                    # when we run out of frames, release the video capture object and break the loop
                    cap.release()
                    break
            # delete the video to free up space for the next
            self.fm.local_delete(vid_path)

    def generate_filename(self, projectID, v_file, frame_count, fps):
        """
        -creates a filename in the format <parent-projectID>_<parent-video-name>_<step(min)>_<frame>_<hr-min-sec.ms>.jpg
        :return: {str} <filename>
        """
        time = frame_count / fps
        hrs = int(time // 3600)
        mins = int(time // 60 % 60)
        secs = round(time % 60, 2)
        time = '{:02d}-{:02d}-{:05.2f}'.format(hrs, mins, secs)
        filename = '{}_{}_{}_{}_{}.jpg'.format(projectID, v_file.split('.')[0], self.timestep, frame_count, time)
        return filename

    def upload_imgs(self):
        """
        upload the results directory to dropbox
        """
        self.fm.upload(self.fm.results_dir)

    def cleanup(self):
        """
        Not yet implemented. Deletes any leftover local directories.
        """
        pass



class FileManager:

    def __init__(self, timestep, subset):
        """
        This class handles various specialized file operations and utilities, including:
            - setting the local_root and cloud_root paths
            - setting the relative paths to data_dir, results_dir, and the master logfile
            - building the local file structure
            - downloading from dropbox (FileManager.download)
            - uploading to dropbox (FileManager.upload)
            - listing dropbox directory contents (FileManager.ls_cloud)
            - deleting local files / directories (FileManager.local_delete)
        The FileManager begins by establishing local and cloud root directories. Below these root directories, the local
        and cloud filesystem structures will be identical. As such, most methods of this class expect an "rpath"
        argument (short for "relative path"), containing the path to a file/directory relative to either root.
        """
        print('initializing file manager')
        self.local_root = Path.home() / 'BioSci-McGrath' / 'Apps' / 'CichlidPiData'
        self.cloud_root = PurePosixPath('cichlidVideo:') / 'BioSci-McGrath' / 'Apps' / 'CichlidPiData'
        self.data_dir = PurePath('__ProjectData')
        self.results_dir = PurePath(f'__TrainingData/CichlidDetection/Training2021/Images_{timestep}/{subset}')
        self.logfile = self.results_dir / 'parsing.log'

        self.mkdir_local(self.data_dir)
        self.mkdir_local(self.results_dir)

    def mkdir_local(self, rpath):
        """
        create a local directory at the location "rpath", relative to the local root
        """
        p = self.local_root / rpath
        if not p.exists():
            p.mkdir(parents=True)

    def download(self, rpath):
        """
        download the file or directory located at cloud_root/rpath to local_root/rpath
        """
        sp.run(['rclone', 'copyto', self.cloud_root / rpath, self.local_root / rpath])

    def upload(self, rpath):
        """
        upload the file or directory located at local_root/rpath to cloud_root/rpath
        """
        sp.run(['rclone', 'copyto', self.local_root / rpath, self.cloud_root / rpath])

    def ls_cloud(self, rpath, dirs_only=False):
        """
        list the contents (files and directories) of a dropbox directory. Set dirs_only=True to list only directories
        """
        if dirs_only:
            return sp.run(['rclone', 'lsf', '--dirs-only', self.cloud_root / rpath], capture_output=True,
                          encoding='utf-8').stdout.split()
        else:
            return sp.run(['rclone', 'lsf', self.cloud_root / rpath], capture_output=True,
                          encoding='utf-8').stdout.split()

    def local_delete(self, rpath):
        """
        Delete a the file or directory located at local_root/rpath
        """
        p = self.local_root / rpath
        if p.is_file():
            os.remove(p)
        elif p.is_dir():
            shutil.rmtree(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subset', type=str, default='all',
                        help='Enter desired analysis subset (such as rock, sand, or all), or a custom search string')
    parser.add_argument('-t', '--timestep', type=int, default=15,
                        help='Enter the desired time interval (in minutes) between captured frames')
    parser.add_argument('-y', '--minyear', type=int, default=2019,
                        help='Only parse project created within or after this year')
    parser.add_argument('-m', '--minmonth', type=int, default=1,
                        help='Only parse project created within or after this month')
    args = parser.parse_args()

    vp = VideoParser(timestep=args.timestep, subset=args.subset, min_year=args.minyear, min_month=args.minmonth)
    vp.execute()
