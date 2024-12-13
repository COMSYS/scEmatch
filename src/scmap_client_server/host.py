import os
import signal
import time
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Iterator

from Pyfhel import PyCtxt
import multiprocessing as mp

from src.scmap_client_server.constants import COMPLETE_RUNTIME


class Benchmark(object):
    def __init__(self, queue: mp.Queue, entity: str, key: str):
        self.queue = queue
        self.entity = entity
        self.key = key
        self.start = time.time()

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_value is None:
            self.queue.put_nowait((self.entity, self.key, time.time() - self.start))


class Host(ABC):
    def __init__(self, name, write_directory, benchmark_collector):
        self.name = name
        self.write_directory = write_directory
        self.benchmark_collector = benchmark_collector

    @abstractmethod
    def run(self, method: str):
        pass

    def _run(self, method: str):
        # reset signal handler to default (necessary for slurm)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with self.benchmark(COMPLETE_RUNTIME):
            self.run(method)
        self.benchmark_collector.post_hook(self)

    def start_as_process(self, method: str) -> mp.Process:
        proc = mp.Process(target=self._run, args=(method,), daemon=False)
        proc.start()
        return proc

    def create_directory(self):
        if not os.path.exists(self.write_directory):
            os.makedirs(self.write_directory)

    def write_bytes(self, file, byte_strings: Iterator[bytes] | list[bytes]):
        with open(file, 'wb') as f:
            for byte_string in byte_strings:
                serialized_byte_string = self.prepend_length(byte_string)
                f.write(serialized_byte_string)

    @staticmethod
    def prepend_length(byte_string):
        """
        Prepend length of byte string to itself.
        :param byte_string:
        :return:
        """
        length = len(byte_string).to_bytes(4, byteorder='big')
        return length + byte_string

    @staticmethod
    def read_bytes(file_path: Path, transformer) -> [PyCtxt]:
        """
        Reads all bytes from a file and transforms them into a list of pyfhel ciphertexts.
        :param file_path:
        :param transformer:
        :return:
        """
        byte_strings = []
        with open(file_path, 'rb') as f:
            while True:
                length_byte_string = f.read(4)
                if not length_byte_string:
                    break
                length = int.from_bytes(length_byte_string, byteorder='big')
                byte_string = f.read(length)
                byte_strings.append(transformer(byte_string))
        return byte_strings

    @staticmethod
    def read_bytes_generator_parallel(file_path, chunk_size) -> Iterator[bytes]:
        with open(file_path, 'rb') as f:
            while True:
                results = []
                while len(results) < chunk_size:
                    length_byte_string = f.read(4)
                    if not length_byte_string:
                        if len(results) > 0:
                            yield results
                        return
                    length = int.from_bytes(length_byte_string, byteorder='big')
                    byte_string = f.read(length)
                    results.append(byte_string)
                yield results

    @staticmethod
    def read_array(file_path: Path):
        array = np.load(file_path)
        assert isinstance(array, np.ndarray)
        return array

    @staticmethod
    def get_size_of_files(directory, files: list[str] | set[str]) -> int:
        """
        Returns file sizes (sum) of selected files contained in ``file_list``.
        :param directory:
        :param files: Container (list or set) containing file names with specified file extensions.
        :return:
        """
        size = 0
        for file in files:
            path_object = Path(directory, file)
            size += path_object.stat().st_size
        return size

    def add_benchmark_result(self, key, value):
        self.benchmark_collector.queue.put_nowait((self.name, key, value))

    def benchmark(self, key):
        return Benchmark(self.benchmark_collector.queue, self.name, key)
