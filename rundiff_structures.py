import os
import subprocess
import shutil
import numpy as np
from multiprocessing import Pool
import time
import h5py
import glob

#paths
ROOT = os.path.abspath(".")
input_data = os.path.join(ROOT, "input_data")
output_data = os.path.join(ROOT, "output_data")
sim_files = os.path.join(input_data, "sim")

#path to gromacs
gmx = '/home/spidocstester/MolDStruct/bin'

#paths to files
MDP = os.path.join(sim_files, "exp.mdp")
ATOMIC_DATA = os.path.join(sim_files, "Atomic_data")
SAMPLED_PDBS_DIR = os.path.join(sim_files, "sampled_pdbs")

#simulation settings
N_REPS = 50
N_INTENSITIES = 20
intensities = np.logspace(10, 13, N_INTENSITIES)
N_cores = 15


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def save_h5(f, name, data, dtype=np.float32):
    f.create_dataset(name, data=data.astype(dtype),
                     compression="gzip", compression_opts=4)

#make h5 file
def process_to_h5(gro, trr, out_file, intensity, output_path, pdb_path = None):
    import MDAnalysis as mda
    
    u = mda.Universe(gro, trr)
    atoms = u.select_atoms("all")

    u.trajectory[0]
    pos0 = atoms.positions.copy()
    vel0 = atoms.velocities.copy()

    u.trajectory[-1]
    posf = atoms.positions.copy()
    velf = atoms.velocities.copy()

    mass         = atoms.masses.copy()
    displacement = posf - pos0
    disp_mag     = np.linalg.norm(displacement, axis=1)

    sim_out = os.path.join(output_path, "simulation_output")

    charge_data  = np.loadtxt(os.path.join(sim_out, "mean_charge_vs_time.txt"))
    charge_times = charge_data[:, 0].astype(np.float32)
    charge_mean  = charge_data[:, 1].astype(np.float32)

    charges_data = np.loadtxt(os.path.join(sim_out, "charges.txt"))
    atom_ids     = charges_data[:, 0].astype(np.int32)
    final_charge = charges_data[:, 1].astype(np.float32)

    pulse_data  = np.loadtxt(os.path.join(sim_out, "pulse_profile.txt"))
    pulse_times = pulse_data[:, 0].astype(np.float32)
    pulse_flux  = pulse_data[:, 1].astype(np.float32)

    stats = np.zeros(5, dtype=np.float32)
    stats_file = os.path.join(sim_out, "procces_statistics.txt")
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            lines = [l.strip() for l in f
                     if l.strip() and not l.startswith("|") and not l.startswith("Processes")]
        if lines:
            vals = lines[0].split()
            stats[:len(vals)] = [float(v) for v in vals]

    with h5py.File(out_file, 'w') as f:
        save_h5(f, "displacement",      displacement)
        save_h5(f, "disp_mag",          disp_mag)
        save_h5(f, "initial_position",  pos0)
        save_h5(f, "final_position",    posf)
        save_h5(f, "initial_velocity",  vel0)
        save_h5(f, "final_velocity",    velf)
        save_h5(f, "mass",              mass)
        save_h5(f, "charge_time",       charge_times)
        save_h5(f, "mean_charge",       charge_mean)
        save_h5(f, "final_charge",      final_charge)
        save_h5(f, "pulse_time",        pulse_times)
        save_h5(f, "pulse_flux",        pulse_flux)
        save_h5(f, "proc_stats",        stats)
        f.create_dataset("atom_ids",     data=atom_ids,      compression="gzip")
        f.create_dataset("atom_indices", data=atoms.indices, compression="gzip")
        f.attrs["intensity"]   = intensity
        f.attrs["proc_labels"] = "auger,fluorescence,photoionization,charge_transfer,unknown"
        f.attrs["source_pdb"] = os.path.basename(pdb_path)


def set_parameters(mdp_path, nsteps=None, time_step=None, pulse_peak=None,
                   num_photons=None, sigma=None, FWHM=None, focus=None,
                   energy=None, charge_transfer=None, autostop=None,
                   autostop_limit=None, logging=None, ionize=None,
                   gen_vel=None, gen_temp=None, log_frequency=None,
                   set_charges=None, rc=None, gen_seed=None):

    if FWHM is not None:
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

    not_none_names = [
        k for k, v in locals().items()
        if v is not None and k not in {"FWHM", "self", "mdp_path"}
    ]
    not_none_values = [
        v for k, v in locals().items()
        if v is not None and k not in {"FWHM", "self", "mdp_path", "not_none_names"}
    ]

    var_mdp = {
        "nsteps": "nsteps", "time_step": "dt", "pulse_peak": "userreal1",
        "num_photons": "userreal2", "sigma": "userreal3", "focus": "userreal4",
        "energy": "userreal5", "charge_transfer": "userint2", "autostop": "userint3",
        "autostop_limit": "userreal6", "logging": "userint5", "ionize": "userint1",
        "gen_vel": "gen-vel", "gen_temp": "gen_temp", "set_charges": "userint9",
        "gen_seed": "gen_seed",
    }

    def change_line(split_line, line, parameter, value):
        if parameter in split_line:
            return f"{parameter}                   = {value}; \n"
        return line

    with open(mdp_path, "r") as f:
        lines = f.readlines()

    with open(mdp_path, "w") as f:
        for line in lines:
            if line.strip().startswith(";") or line.strip() == "":
                f.write(line)
                continue
            split_line = line.split()
            for name, value in zip(not_none_names, not_none_values):
                if name == "log_frequency":
                    for param in ["nstxout", "nstfout", "nstvout", "nstenergy", "nstlog", "xtc_precision"]:
                        if param in split_line:
                            line = change_line(split_line, line, param, value)
                elif name == "rc":
                    for param in ["rlist", "rcoulomb", "rvdw"]:
                        if param in split_line:
                            line = change_line(split_line, line, param, value)
                else:
                    line = change_line(split_line, line, var_mdp[name], value)
            f.write(line)


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")

def run_pdb2gmx(pdb, top, gro, itp):
    return run([os.path.join(gmx, "pdb2gmx"), "-f", pdb, "-p", top, "-o", gro,
                "-i", itp, "-water", "tip3p", "-ff", "charmm36-mar2019-Fe-S", "-ignh"])

def run_grompp(mdp, gro, top, tpr, mdout):
    return run([os.path.join(gmx, "grompp"), "-f", mdp, "-c", gro, "-p", top,
                "-o", tpr, "-po", mdout, "-maxwarn", "5"])

def run_mdrun(tpr, output_path):
    sim_out = os.path.join(output_path, "simulation_output")
    os.makedirs(sim_out, exist_ok=True)
    trr = os.path.join(sim_out, "exp.trr")
    xtc = os.path.join(sim_out, "exp.xtc")
    cmd = [os.path.join(gmx, "mdrun"), "-s", tpr, "-o", trr, "-x", xtc,
           "-nt", "1", "-v", "-ionize"]
    with open(os.path.join(output_path, "mdrun.stdout"), "w") as so, \
         open(os.path.join(output_path, "mdrun.stderr"), "w") as se:
        result = subprocess.run(cmd, cwd=output_path, stdout=so, stderr=se,
                                text=True, errors="replace")
    return result, trr, xtc


def run_sim(params):
    pdb_path, mdp_path, output_path, intensity = params

    done_flag = os.path.join(output_path, "DONE")
    if os.path.exists(done_flag):
        return

    log(f"Starting: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    top     = os.path.join(output_path, "topol.top")
    gro     = os.path.join(output_path, "conf.gro")
    itp     = os.path.join(output_path, "posre.itp")
    tpr     = os.path.join(output_path, "explode.tpr")
    mdout   = os.path.join(output_path, "mdout.mdp")
    sim_mdp = os.path.join(output_path, "sim.mdp")
    h5_out  = os.path.join(output_path, "results.h5")

    shutil.copy(mdp_path, sim_mdp)

    atomic_dst = os.path.join(output_path, "Atomic_data")
    if not os.path.exists(atomic_dst):
        os.symlink(ATOMIC_DATA, atomic_dst)

    if run_pdb2gmx(pdb_path, top, gro, itp).returncode != 0:
        log(f"pdb2gmx failed: {output_path}")
        return
#simulation parameters
    set_parameters(sim_mdp, num_photons=intensity, FWHM=0.01,
                   charge_transfer=1, nsteps=150000, log_frequency=50000)

    if run_grompp(sim_mdp, gro, top, tpr, mdout).returncode != 0:
        log(f"grompp failed: {output_path}")
        return

    result, trr, xtc = run_mdrun(tpr, output_path)
    if result.returncode != 0:
        log(f"mdrun failed: {output_path}")
        return

    try:
        process_to_h5(gro, trr, h5_out, intensity, output_path, pdb_path)
        os.remove(trr)
        sim_out = os.path.join(output_path, "simulation_output")
        for fname in ["electronic_transition_log.txt", "electron_data.txt",
                      "mean_charge_vs_time.txt", "charges.txt",
                      "pulse_profile.txt", "procces_statistics.txt"]:
            fpath = os.path.join(sim_out, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
    except Exception as e:
        log(f"H5 processing failed: {output_path} — {e}")
        return

    with open(done_flag, "w") as f:
        f.write("OK\n")

    log(f"Finished: {output_path}")


if __name__ == "__main__":
    PDB_FILES = sorted(glob.glob(os.path.join(SAMPLED_PDBS_DIR, "start_*.pdb")))
    
    log(f"Found {len(PDB_FILES)} structure files")
    
    # Single output directory for all structures
    sim_output = os.path.join(output_data, "sim_results_sampled_structures")
    
    params = []
    for intensity in intensities:
        intensity_dir = os.path.join(sim_output, f"I_{intensity:.2e}")
        for rep_idx, pdb_file in enumerate(PDB_FILES):
            output_path = os.path.join(intensity_dir, f"replica_{rep_idx:02d}")
            params.append((pdb_file, MDP, output_path, intensity))

    log(f"Total structures: {len(PDB_FILES)}")
    log(f"Total runs: {len(params)} ({N_INTENSITIES} × {len(PDB_FILES)})")
    log(f"Using {N_cores} workers")

    with Pool(N_cores) as pool:
        pool.map(run_sim, params)

    log("All simulations complete.")