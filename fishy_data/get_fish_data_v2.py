import cv2
import subprocess as sp
from pathlib import Path, PurePath, PurePosixPath
from datetime import datetime as dt
import shutil
import os
import argparse
from tqdm import tqdm

class VideoParser:

    def __init__(self, timestep, subset, min_year=2019, min_month=1):
        self.timestep, self.subset, self.min_year, self.min_month = timestep, subset, min_year, min_month
        self.fm = FileManager(timestep, subset)
        self.fm.mkdir_local(self.fm.data_dir)

    def execute(self):
        # execute all steps in video parsing
        pids = self.get_pids()
        for pid in pids:
            self.parse_project(pid)
        self.upload_imgs()

    def get_pids(self):
        # gather all project ID's that meet the subset, min_year, and min_month conditions
        print('determining valid projects')
        all_pids = self.fm.ls_cloud(self.fm.data_dir, dirs_only=True)

        if self.subset == 'rock':
            search_strings = ['rock', 'mz', 'rs', 'kl']
        elif self.subset == 'sand':
            search_strings = ['sand', 'ti', 'mc', 'cv']
        else:
            search_strings = [self.subset]

        valid_pids = []
        for pid in tqdm(all_pids):
            if (self.subset == 'all') or (any(s in pid.lower() for s in search_strings)):
                year, month = self.check_creation_date(pid)
                if (year >= self.min_month) and (month >= self.min_month):
                    valid_pids.append(pid.strip('/'))
        print(f'{len(valid_pids)} valid projects located')
        return valid_pids

    def check_creation_date(self, pid):
        # check a project's creation date robustly using the logfile
        rpath = self.fm.data_dir / pid / 'Logfile.txt'
        if len(self.fm.ls_cloud(rpath)) > 0:
            self.fm.download(rpath)
            lpath = (self.fm.local_root / rpath)
            with open(lpath) as f:
                for line in f:
                    if line.startswith('MasterRecordInitialStart:'):
                        t = line.split(': ')[-1].strip()
                        t = dt.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
            self.fm.local_delete(rpath)
            return t.year, t.month
        else:
            return 0, 0

    def parse_project(self, pid):
        # parse all videos from a given project (pid) into images
        print(f'parsing videos for {pid}')
        vid_dir = self.fm.data_dir / pid / 'Videos'
        vid_names = [n for n in self.fm.ls_cloud(vid_dir) if (('.mp4' in n) or ('.h264' in n))]
        for vname in vid_names:
            vid_path = vid_dir / vname
            self.fm.download(vid_path)
            cap = cv2.VideoCapture(str(self.fm.local_root / vid_path))
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            frame_step = int(self.timestep * 60 * fps)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    img_name = self.generate_filename(pid, vname, count, fps)
                    img_path = self.fm.local_root / self.fm.results_dir / img_name
                    cv2.imwrite(str(img_path), frame)
                    count += frame_step
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                else:
                    cap.release()
                    break
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
        # upload the results directory
        self.fm.upload(self.fm.results_dir)


class FileManager:

    def __init__(self, timestep, subset):
        print('initializing file manager')
        self.local_root = Path.home() / 'BioSci-McGrath' / 'Apps' / 'CichlidPiData'
        self.cloud_root = PurePosixPath('cichlidVideo:') / 'BioSci-McGrath' / 'Apps' / 'CichlidPiData'
        self.data_dir = PurePath('__ProjectData')
        self.results_dir = PurePath(f'__TrainingData/CichlidDetection/Training2021/Images_{timestep}/{subset}')
        self.logfile = self.results_dir / 'parsing.log'

        self.mkdir_local(self.data_dir)
        self.mkdir_local(self.results_dir)

    def mkdir_local(self, rpath):
        p = self.local_root / rpath
        if not p.exists():
            p.mkdir(parents=True)

    def download(self, rpath):
        sp.run(['rclone', 'copyto', self.cloud_root / rpath, self.local_root / rpath])

    def upload(self, rpath):
        sp.run(['rclone', 'copyto', self.local_root / rpath, self.cloud_root / rpath])

    def ls_cloud(self, rpath, dirs_only=False):
        if dirs_only:
            return sp.run(['rclone', 'lsf', '--dirs-only', self.cloud_root / rpath], capture_output=True,
                          encoding='utf-8').stdout.split()
        else:
            return sp.run(['rclone', 'lsf', self.cloud_root / rpath], capture_output=True,
                          encoding='utf-8').stdout.split()

    def local_delete(self, rpath):
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
    args = parser.parse_args()

    vp = VideoParser(timestep=args.timestep, subset=args.subset)
    vp.execute()
