"""
Data loader for NE2001 parameter files

Loads all NE2001 input parameter files and converts them to PyTorch tensors
for use in the model components.
"""

import os
import torch
import numpy as np
from typing import Dict, Any


class NE2001Data:
    """
    Container for NE2001 model parameters loaded from input files.

    All parameters are stored as PyTorch tensors on the specified device.
    """

    def __init__(self, data_dir: str = None, device: str = 'cpu'):
        """
        Load NE2001 parameter files into PyTorch tensors.

        Args:
            data_dir: Directory containing .inp and .dat files.
                     If None, uses pygedm package data directory.
            device: Device to load tensors on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        if data_dir is None:
            # Use pygedm package data directory
            try:
                from pkg_resources import resource_filename
                data_dir = resource_filename('pygedm', '')
            except:
                # Fallback to relative path
                import torchgedm
                package_dir = os.path.dirname(os.path.dirname(torchgedm.__file__))
                data_dir = os.path.join(package_dir, 'pygedm')

        self.data_dir = data_dir

        # Load all parameter files
        self.load_gal01()
        self.load_ne_arms()
        self.load_ne_gc()
        self.load_nelism()
        self.load_clumps()
        self.load_voids()

    def load_gal01(self):
        """Load gal01.inp - large-scale component parameters"""
        filepath = os.path.join(self.data_dir, 'gal01.inp')

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse parameters (skip header lines)
        params = {}
        for line in lines[1:]:  # Skip first line (header)
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split on colon if present
            if ':' in line:
                parts = line.split(':')
                name = parts[0].strip()
                value_str = parts[1].split()[0]  # Get first value
                try:
                    value = float(value_str)
                    params[name] = value
                except ValueError:
                    # Handle weight line differently
                    if name == 'gal01.inp = input parameters for large-scale components of NE2001 30 June \'02':
                        continue

        # Parse weights from line 2
        weight_line = lines[1].strip()
        weights = weight_line.split()[:7]  # First 7 values are weights
        self.wg1 = torch.tensor(float(weights[0]), device=self.device)
        self.wg2 = torch.tensor(float(weights[1]), device=self.device)
        self.wga = torch.tensor(float(weights[2]), device=self.device)
        self.wggc = torch.tensor(float(weights[3]), device=self.device)
        self.wglism = torch.tensor(float(weights[4]), device=self.device)
        self.wgcN = torch.tensor(float(weights[5]), device=self.device)
        self.wgvN = torch.tensor(float(weights[6]), device=self.device)

        # Store component 1 parameters (outer disk)
        self.n1h1 = torch.tensor(params['n1h1'], device=self.device)
        self.h1 = torch.tensor(params['h1'], device=self.device)
        self.A1 = torch.tensor(params['A1'], device=self.device)
        self.F1 = torch.tensor(params['F1'], device=self.device)

        # Store component 2 parameters (inner disk)
        self.n2 = torch.tensor(params['n2'], device=self.device)
        self.h2 = torch.tensor(params['h2'], device=self.device)
        self.A2 = torch.tensor(params['A2'], device=self.device)
        self.F2 = torch.tensor(params['F2'], device=self.device)

        # Store spiral arm parameters
        self.na = torch.tensor(params['na'], device=self.device)
        self.ha = torch.tensor(params['ha'], device=self.device)
        self.wa = torch.tensor(params['wa'], device=self.device)
        self.Aa = torch.tensor(params['Aa'], device=self.device)
        self.Fa = torch.tensor(params['Fa'], device=self.device)

        # Store individual arm parameters (5 arms)
        self.narm = torch.tensor([params[f'narm{i}'] for i in range(1, 6)], device=self.device)
        self.warm = torch.tensor([params[f'warm{i}'] for i in range(1, 6)], device=self.device)
        self.harm = torch.tensor([params[f'harm{i}'] for i in range(1, 6)], device=self.device)
        self.farm = torch.tensor([params[f'farm{i}'] for i in range(1, 6)], device=self.device)

    def load_ne_arms(self):
        """Load ne_arms_log_mod.inp - spiral arm geometry"""
        filepath = os.path.join(self.data_dir, 'ne_arms_log_mod.inp')

        # Read data (skip first 2 header lines)
        data = np.loadtxt(filepath, skiprows=2)

        # Each row is: a, rmin, thmin, extent
        self.arm_a = torch.tensor(data[:, 0], device=self.device, dtype=torch.float32)
        self.arm_rmin = torch.tensor(data[:, 1], device=self.device, dtype=torch.float32)
        self.arm_thmin = torch.tensor(data[:, 2], device=self.device, dtype=torch.float32)
        self.arm_extent = torch.tensor(data[:, 3], device=self.device, dtype=torch.float32)

    def load_ne_gc(self):
        """Load ne_gc.inp - Galactic center parameters"""
        filepath = os.path.join(self.data_dir, 'ne_gc.inp')

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Line 2: xgc, ygc, zgc (comma-separated, with comments after tab)
        line2 = lines[1].split('\t')[0]  # Take part before tab (removes comments)
        gc_center = [float(x.strip()) for x in line2.split(',')[:3]]
        self.xgc = torch.tensor(gc_center[0], device=self.device)
        self.ygc = torch.tensor(gc_center[1], device=self.device)
        self.zgc = torch.tensor(gc_center[2], device=self.device)

        # Line 3: rgc (value before tab)
        self.rgc = torch.tensor(float(lines[2].split('\t')[0].strip()), device=self.device)

        # Line 4: hgc (value before tab)
        self.hgc = torch.tensor(float(lines[3].split('\t')[0].strip()), device=self.device)

        # Line 5: negc0 (value before tab)
        self.negc0 = torch.tensor(float(lines[4].split('\t')[0].strip()), device=self.device)

        # Line 6: Fgc0 (value before tab)
        self.Fgc0 = torch.tensor(float(lines[5].split('\t')[0].strip()), device=self.device)

    def load_nelism(self):
        """Load nelism.inp - Local ISM parameters"""
        filepath = os.path.join(self.data_dir, 'nelism.inp')

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # LDR (Low Density Region) - lines 2-4
        ldr_abc = [float(x) for x in lines[1].split()[:3]]
        ldr_xyz = [float(x) for x in lines[2].split()[:3]]
        ldr_params = [float(x) for x in lines[3].split()[:3]]

        self.aldr = torch.tensor(ldr_abc[0], device=self.device)
        self.bldr = torch.tensor(ldr_abc[1], device=self.device)
        self.cldr = torch.tensor(ldr_abc[2], device=self.device)
        self.xldr = torch.tensor(ldr_xyz[0], device=self.device)
        self.yldr = torch.tensor(ldr_xyz[1], device=self.device)
        self.zldr = torch.tensor(ldr_xyz[2], device=self.device)
        self.thetaldr = torch.tensor(ldr_params[0], device=self.device)
        self.neldr = torch.tensor(ldr_params[1], device=self.device)
        self.Fldr = torch.tensor(ldr_params[2], device=self.device)

        # LSB (Local Super Bubble) - lines 5-7
        lsb_abc = [float(x) for x in lines[4].split()[:3]]
        lsb_xyz = [float(x) for x in lines[5].split()[:3]]
        lsb_params = [float(x) for x in lines[6].split()[:3]]

        self.alsb = torch.tensor(lsb_abc[0], device=self.device)
        self.blsb = torch.tensor(lsb_abc[1], device=self.device)
        self.clsb = torch.tensor(lsb_abc[2], device=self.device)
        self.xlsb = torch.tensor(lsb_xyz[0], device=self.device)
        self.ylsb = torch.tensor(lsb_xyz[1], device=self.device)
        self.zlsb = torch.tensor(lsb_xyz[2], device=self.device)
        self.thetalsb = torch.tensor(lsb_params[0], device=self.device)
        self.nelsb = torch.tensor(lsb_params[1], device=self.device)
        self.Flsb = torch.tensor(lsb_params[2], device=self.device)

        # LHB (Local Hot Bubble) - lines 8-10
        lhb_abc = [float(x) for x in lines[7].split()[:3]]
        lhb_xyz = [float(x) for x in lines[8].split()[:3]]
        lhb_params = [float(x) for x in lines[9].split()[:3]]

        self.alhb = torch.tensor(lhb_abc[0], device=self.device)
        self.blhb = torch.tensor(lhb_abc[1], device=self.device)
        self.clhb = torch.tensor(lhb_abc[2], device=self.device)
        self.xlhb = torch.tensor(lhb_xyz[0], device=self.device)
        self.ylhb = torch.tensor(lhb_xyz[1], device=self.device)
        self.zlhb = torch.tensor(lhb_xyz[2], device=self.device)
        self.thetalhb = torch.tensor(lhb_params[0], device=self.device)
        self.nelhb = torch.tensor(lhb_params[1], device=self.device)
        self.Flhb = torch.tensor(lhb_params[2], device=self.device)

        # Loop I - lines 11-13
        lpI_xyz = [float(x) for x in lines[10].split()[:3]]
        lpI_r = [float(x) for x in lines[11].split()[:2]]
        lpI_params = [float(x) for x in lines[12].split()[:4]]

        self.xlpI = torch.tensor(lpI_xyz[0], device=self.device)
        self.ylpI = torch.tensor(lpI_xyz[1], device=self.device)
        self.zlpI = torch.tensor(lpI_xyz[2], device=self.device)
        self.rlpI = torch.tensor(lpI_r[0], device=self.device)
        self.drlpI = torch.tensor(lpI_r[1], device=self.device)
        self.nelpI = torch.tensor(lpI_params[0], device=self.device)
        self.dnelpI = torch.tensor(lpI_params[1], device=self.device)
        self.FlpI = torch.tensor(lpI_params[2], device=self.device)
        self.dFlpI = torch.tensor(lpI_params[3], device=self.device)

    def load_clumps(self):
        """Load neclumpN.NE2001.dat - discrete clump data"""
        filepath = os.path.join(self.data_dir, 'neclumpN.NE2001.dat')

        # Read data, skip header, handle mixed types
        data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) >= 9:
                    # Extract numeric columns: index, l, b, nec, Fc, dc, rc, edge
                    row = [
                        float(parts[0]),  # index
                        float(parts[1]),  # l
                        float(parts[2]),  # b
                        float(parts[3]),  # nec
                        float(parts[4]),  # Fc
                        float(parts[5]),  # dc
                        float(parts[6]),  # rc
                        float(parts[7]),  # edge
                    ]
                    data.append(row)

        data = np.array(data)

        # Store as tensors (shape: [n_clumps, features])
        self.clump_l = torch.tensor(data[:, 1], device=self.device, dtype=torch.float32)
        self.clump_b = torch.tensor(data[:, 2], device=self.device, dtype=torch.float32)
        self.clump_nec = torch.tensor(data[:, 3], device=self.device, dtype=torch.float32)
        self.clump_Fc = torch.tensor(data[:, 4], device=self.device, dtype=torch.float32)
        self.clump_dc = torch.tensor(data[:, 5], device=self.device, dtype=torch.float32)
        self.clump_rc = torch.tensor(data[:, 6], device=self.device, dtype=torch.float32)
        self.clump_edge = torch.tensor(data[:, 7], device=self.device, dtype=torch.float32)

        self.n_clumps = len(self.clump_l)

    def load_voids(self):
        """Load nevoidN.NE2001.dat - discrete void data"""
        filepath = os.path.join(self.data_dir, 'nevoidN.NE2001.dat')

        # Read data, skip header
        data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) >= 11:
                    # Extract numeric columns
                    row = [
                        float(parts[0]),   # index
                        float(parts[1]),   # l
                        float(parts[2]),   # b
                        float(parts[3]),   # dv
                        float(parts[4]),   # nev
                        float(parts[5]),   # Fv
                        float(parts[6]),   # aav
                        float(parts[7]),   # bbv
                        float(parts[8]),   # ccv
                        float(parts[9]),   # thvy
                        float(parts[10]),  # thvz
                        float(parts[11]),  # edge
                    ]
                    data.append(row)

        data = np.array(data)

        # Store as tensors (shape: [n_voids, features])
        self.void_l = torch.tensor(data[:, 1], device=self.device, dtype=torch.float32)
        self.void_b = torch.tensor(data[:, 2], device=self.device, dtype=torch.float32)
        self.void_dv = torch.tensor(data[:, 3], device=self.device, dtype=torch.float32)
        self.void_nev = torch.tensor(data[:, 4], device=self.device, dtype=torch.float32)
        self.void_Fv = torch.tensor(data[:, 5], device=self.device, dtype=torch.float32)
        self.void_aav = torch.tensor(data[:, 6], device=self.device, dtype=torch.float32)
        self.void_bbv = torch.tensor(data[:, 7], device=self.device, dtype=torch.float32)
        self.void_ccv = torch.tensor(data[:, 8], device=self.device, dtype=torch.float32)
        self.void_thvy = torch.tensor(data[:, 9], device=self.device, dtype=torch.float32)
        self.void_thvz = torch.tensor(data[:, 10], device=self.device, dtype=torch.float32)
        self.void_edge = torch.tensor(data[:, 11], device=self.device, dtype=torch.float32)

        self.n_voids = len(self.void_l)

    def to(self, device):
        """Move all tensors to specified device"""
        self.device = torch.device(device)

        # Move all tensor attributes to new device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))

        return self

    def __repr__(self):
        return (f"NE2001Data(device={self.device}, "
                f"n_clumps={self.n_clumps}, n_voids={self.n_voids})")
