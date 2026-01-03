"""
memory_manager.py - Memory management untuk matriks besar
"""

import numpy as np
import tempfile
import gc
from pathlib import Path
from typing import Optional


class Memory_Manager:
    """
    Manage memory-mapped arrays for handling large distance matrices.
    
    Creates temporary files to back large matrices on disk, reducing peak
    RAM usage during expensive topological computations.
    
    Attributes
    ----------
    temp_dir : Path
        Temporary directory for memory-mapped files.
    _files : list
        Paths to created memory-mapped files.
    _arrays : list
        References to memory-mapped array objects.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize Memory_Manager.
        
        Parameters
        ----------
        base_dir : str, optional
            Custom base directory. If None, uses system temp.
        """
        self.temp_dir = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix='entrap_'))
        if base_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._files = []
        self._arrays = []

    def create(self, shape: tuple, dtype=np.float64, name: str = None) -> np.memmap:
        """
        Create a memory-mapped array.
        
        Parameters
        ----------
        shape : tuple
            Shape of array.
        dtype : type, default=np.float64
            Data type.
        name : str, optional
            Prefix for temporary file name.
        
        Returns
        -------
        ndarray or memmap
            Memory-mapped array.
        """
        fname = self.temp_dir / (f'{name}.dat' if name else f'memmap_{len(self._files)}.dat')
        self._files.append(fname)
        mmap = np.memmap(str(fname), dtype=dtype, mode='w+', shape=shape)
        self._arrays.append(mmap)
        return mmap

    def cleanup(self):
        """
        Release memory-mapped arrays and delete temporary files.
        
        Safely flushes and deletes all backing files. Called automatically
        on object deletion.
        """
        for arr in self._arrays:
            try:
                arr.flush()
                del arr
            except:
                pass
        self._arrays.clear()
        gc.collect()

        for fname in self._files:
            try:
                if fname.exists():
                    fname.unlink()
            except:
                pass
        self._files.clear()

        try:
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
        except:
            pass

    def __del__(self):
        self.cleanup()