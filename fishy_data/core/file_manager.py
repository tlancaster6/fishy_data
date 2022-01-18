import shutil
import os
from pathlib import PurePath, PurePosixPath
import subprocess as sp
from fishy_data.definitions import PACKAGE_DIR

class FileManager:

    def __init__(self):
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
        self.package_dir = PurePath(PACKAGE_DIR)
        self.local_root = self.package_dir / 'data'
        self.cloud_root = PurePosixPath('cichlidVideo:') / 'BioSci-McGrath' / 'Apps' / 'CichlidPiData'

    def add_path(self, name, path):
        setattr(self, name, PurePath(path))

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
