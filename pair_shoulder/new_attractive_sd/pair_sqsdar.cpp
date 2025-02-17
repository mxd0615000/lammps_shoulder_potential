/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jorge Ram�rez (jorge.ramirez@upm.es)
------------------------------------------------------------------------- */

#include "pair_sqsdar.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "integrate.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairSqsdar::PairSqsdar(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  writedata = 1;
  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairSqsdar::~PairSqsdar()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(c0);
    memory->destroy(epxn);
    memory->destroy(k0);
    memory->destroy(sigmas);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

// void PairSqsdar::compute(int eflag, int vflag)
// {
//   int i, j, ii, jj, inum, jnum, itype, jtype;
//   double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
//   double rsq, r, r2inv, rinv, forceij, factor_ij, tanhij, r2min, r2max;
//   int *ilist, *jlist, *numneigh, **firstneigh;

//   evdwl = 0.0;
//   if (eflag || vflag)
//     ev_setup(eflag, vflag);
//   else
//     evflag = vflag_fdotr = 0;

//   double **x = atom->x;
//   double **f = atom->f;
//   int *type = atom->type;
//   int nlocal = atom->nlocal;
//   double *special_lj = force->special_lj;
//   int newton_pair = force->newton_pair;

//   inum = list->inum;
//   ilist = list->ilist;
//   numneigh = list->numneigh;
//   firstneigh = list->firstneigh;

//   // loop over neighbors of my atoms

//   for (ii = 0; ii < inum; ii++) {
//     i = ilist[ii];
//     xtmp = x[i][0];
//     ytmp = x[i][1];
//     ztmp = x[i][2];
//     itype = type[i];
//     jlist = firstneigh[i];
//     jnum = numneigh[i];

//     for (jj = 0; jj < jnum; jj++) {
//       j = jlist[jj];
//       factor_ij = special_lj[sbmask(j)];
//       j &= NEIGHMASK;

//       delx = xtmp - x[j][0];
//       dely = ytmp - x[j][1];
//       delz = ztmp - x[j][2];
//       rsq = delx * delx + dely * dely + delz * delz;

//       jtype = type[j];

//       if (rsq < cutsq[itype][jtype]) {

//         r2min = (z0[itype][jtype] - 10 * dwell[itype][jtype]);
//         r2min *= r2min;
//         r2max = (z0[itype][jtype] + 10 * dwell[itype][jtype]);
//         r2max *= r2max;
//         if (rsq < r2min) {
//           // CALCULAR ENERGIA
//           evdwl = -epsilon[itype][jtype];
//           evdwl *= factor_ij;
//         } else if (rsq > r2max)
//           continue;
//         else {

//           r2inv = 1.0 / rsq;
//           r = sqrt(rsq);
//           rinv = 1.0 / r;
//           tanhij = tanh((r - z0[itype][jtype]) / dwell[itype][jtype]);
//           forceij =
//               -0.5 * epsilon[itype][jtype] * (1 - tanhij * tanhij) * rinv / dwell[itype][jtype];
//           fpair = factor_ij * forceij;

//           f[i][0] += delx * fpair;
//           f[i][1] += dely * fpair;
//           f[i][2] += delz * fpair;
//           if (newton_pair || j < nlocal) {
//             f[j][0] -= delx * fpair;
//             f[j][1] -= dely * fpair;
//             f[j][2] -= delz * fpair;
//           }

//           if (eflag) {
//             evdwl = -0.5 * epsilon[itype][jtype] * (1 - tanhij);
//             evdwl *= factor_ij;
//           }
//         }

//         if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
//       }
//     }
//   }

//   if (vflag_fdotr) virial_fdotr_compute();
// }



/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSqsdar::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
  memory->create(sigma, n + 1, n + 1, "pair:sigma");
  memory->create(c0, n + 1, n + 1, "pair:c0");
  memory->create(epxn, n + 1, n + 1, "pair:epxn");
  memory->create(k0, n + 1, n + 1, "pair:k0");
  memory->create(sigmas, n + 1, n + 1, "pair:sigmas");
  memory->create(offset, n + 1, n + 1, "pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSqsdar::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  //cut_global = force->numeric(FLERR,arg[0]);
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSqsdar::coeff(int narg, char **arg)
{
  if (narg < 8 || narg > 9) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  // force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
  // force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);
  double c0_one = utils::numeric(FLERR, arg[4], false, lmp);
  double epxn_one = utils::numeric(FLERR, arg[5], false, lmp);
  double k0_one = utils::numeric(FLERR, arg[6], false, lmp);
  double sigmas_one = utils::numeric(FLERR, arg[7], false, lmp);
  double cut_one = cut_global;
  if (narg == 11) cut_one = utils::numeric(FLERR, arg[8], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      c0[i][j] = c0_one;
      epxn[i][j] = epxn_one;
      k0[i][j] = k0_one;
      sigmas[i][j] = sigmas_one;
      cut[i][j] = cut_one;
      setflag[j][i] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}


void PairSqsdar::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, r2inv, rinv, forceij, factor_lj, tanhij, r2min, r2max;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        r = sqrt(rsq);
        rinv = 1.0 / r;
        tanhij = tanh(k0[itype][jtype] * (r - sigmas[itype][jtype]));
        forceij = epsilon[itype][jtype] *epxn[itype][jtype]* pow((sigma[itype][jtype]*rinv), epxn[itype][jtype]+1) *sigma[itype][jtype] * rinv * rinv - c0[itype][jtype] * epsilon[itype][jtype] * k0[itype][jtype] * (1 - tanhij * tanhij);
        fpair = factor_lj * forceij; // * rinv

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) evdwl = epsilon[itype][jtype] * pow((sigma[itype][jtype]*rinv) ,epxn[itype][jtype]) - c0[itype][jtype] * epsilon[itype][jtype] * (1 - tanhij) - offset[itype][jtype] ;evdwl *= factor_lj;
        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

// /* ----------------------------------------------------------------------
//    init specific to this pair style
// ------------------------------------------------------------------------- */

// void PairSqsdar::init_style()
// {
//   // request regular or rRESPA neighbor list

//   int list_style = NeighConst::REQ_DEFAULT;

//   if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
//     auto respa = dynamic_cast<Respa *>(update->integrate);
//     if (respa->level_inner >= 0) list_style = NeighConst::REQ_RESPA_INOUT;
//     if (respa->level_middle >= 0) list_style = NeighConst::REQ_RESPA_ALL;
//   }
//   neighbor->add_request(this, list_style);

//   // set rRESPA cutoffs

//   if (utils::strmatch(update->integrate_style, "^respa") &&
//       (dynamic_cast<Respa *>(update->integrate))->level_inner >= 0)
//     cut_respa = (dynamic_cast<Respa *>(update->integrate))->cutoff;
//   else
//     cut_respa = nullptr;
// }

// /* ----------------------------------------------------------------------
//    init for one type pair i,j and corresponding j,i
// ------------------------------------------------------------------------- */

// double PairSqsdar::init_one(int i, int j)
// {
//   if (setflag[i][j] == 0) {
//     epsilon[i][j] = mix_distance(epsilon[i][i], epsilon[j][j]);
//     dwell[i][j] = mix_distance(dwell[i][i], dwell[j][j]);
//     epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j], z0[i][i], z0[j][j]);
//     cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
//   }

//   // check interior rRESPA cutoff

//   if (cut_respa && cut[i][j] < cut_respa[3])
//     error->all(FLERR, "Pair cutoff < Respa interior cutoff");

//   // compute I,J contribution to long-range tail correction
//   // count total # of atoms of type I and J via Allreduce

//   if (tail_flag) {
//     int *type = atom->type;
//     int nlocal = atom->nlocal;

//     double count[2], all[2];
//     count[0] = count[1] = 0.0;
//     for (int k = 0; k < nlocal; k++) {
//       if (type[k] == i) count[0] += 1.0;
//       if (type[k] == j) count[1] += 1.0;
//     }
//     MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);

//     //double sig2 = sigma[i][j]*sigma[i][j];
//     //double sig6 = sig2*sig2*sig2;
//     //double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
//     //double rc6 = rc3*rc3;
//     //double rc9 = rc3*rc6;
//     //etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
//     //  sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
//     //ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
//     //  sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
//   }

//   return cut[i][j];
// }

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSqsdar::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  if (offset_flag) {
    // double dr = cut[i][j] - r0[i][j];
    offset[i][j] =
        epsilon[i][j] * pow((sigma[i][j]/cut[i][j]),epxn[i][j]) - c0[i][j]*epsilon[i][j] * (1- tanh(k0[i][j]*(cut[i][j]-sigmas[i][j])));
  } else
    offset[i][j] = 0.0;

  epsilon[j][i] = epsilon[i][j];
  sigma[j][i] = sigma[i][j];
  c0[j][i] = c0[i][j];
  epxn[j][i] = epxn[i][j];
  k0[j][i] = k0[i][j];
  sigmas[j][i] = sigmas[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSqsdar::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&c0[i][j], sizeof(double), 1, fp);
        fwrite(&epxn[i][j], sizeof(double), 1, fp);
        fwrite(&k0[i][j], sizeof(double), 1, fp);
        fwrite(&sigmas[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSqsdar::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  //  fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSqsdar::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      //if (me == 0) fread(&setflag[i][j], sizeof(int), 1, fp);
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          // fread(&z0[i][j], sizeof(double), 1, fp);
          // fread(&dwell[i][j], sizeof(double), 1, fp);
          // fread(&epsilon[i][j], sizeof(double), 1, fp);
          // fread(&cut[i][j], sizeof(double), 1, fp);
          utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &c0[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &epxn[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &k0[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigmas[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&c0[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&epxn[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&k0[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigmas[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSqsdar::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    //fread(&cut_global, sizeof(double), 1, fp);
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    //    fread(&offset_flag,sizeof(int),1,fp);
    //fread(&mix_flag, sizeof(int), 1, fp);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    //fread(&tail_flag, sizeof(int), 1, fp);
    // utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  // MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairSqsdar::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, epsilon[i][i]);
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, sigma[i][i]);
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, c0[i][i]);
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, epxn[i][i]);
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, k0[i][i]);
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, sigmas[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairSqsdar::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g %g\n", i, j, epsilon[i][j], sigma[i][j], c0[i][j], epxn[i][j], k0[i][j], sigmas[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairSqsdar::single(int i, int j, int itype, int jtype, double rsq, double factor_coul,
                            double factor_lj, double &fforce)
{
  double r, r2inv, rinv, tanhij, forceij, philj;
  r2inv = 1.0 / rsq;
  r = sqrt(rsq);
  rinv = 1.0 / r;
  tanhij = tanh(k0[itype][jtype]*(r - sigmas[itype][jtype]));
  // forceij = -0.5 * epsilon[itype][jtype] * (1 - tanhij * tanhij) * rinv / dwell[itype][jtype];
  forceij = epsilon[itype][jtype] *epxn[itype][jtype]* pow((sigma[itype][jtype]*rinv) ,epxn[itype][jtype]+1) *sigma[itype][jtype] * rinv * rinv - c0[itype][jtype] * epsilon[itype][jtype] * k0[itype][jtype] * (1 - tanhij * tanhij);
  //r6inv * lj1[itype][jtype] * r6inv + lj2[itype][jtype] * r*(pow(tanh((r - delta[itype][jtype]) / gamma[itype][jtype]), 2.0) + 1.0);
  fforce = factor_lj * forceij;

  philj = epsilon[itype][jtype] * pow((sigma[itype][jtype]*rinv) ,epxn[itype][jtype]) - c0[itype][jtype] * epsilon[itype][jtype] * (1 - tanhij);
  // philj = -0.5 * epsilon[itype][jtype] * (1 - tanhij);
  //r6inv*lj3[itype][jtype] * r6inv - lj4[itype][jtype] * (tanh((r - delta[itype][jtype]) / gamma[itype][jtype]) - 1.0);
  return factor_lj * philj;
}

/* ---------------------------------------------------------------------- */

void *PairSqsdar::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "epsilon") == 0) return (void *) epsilon;
  if (strcmp(str, "sigmas") == 0) return (void *) sigmas;
  if (strcmp(str, "c0") == 0) return (void *) c0;
  if (strcmp(str, "epxn") == 0) return (void *) epxn;
  if (strcmp(str, "sigma") == 0) return (void *) sigma;
  if (strcmp(str, "k0") == 0) return (void *) k0;
  return NULL;
}
