
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main");

        int test_vector = 1;
        IntVect n_cell{AMREX_D_DECL(64,64,64)};
        int nghost = 1;
        {
            Vector<int> n_cell_v;
            ParmParse pp;
            if (pp.queryarr("n_cell", n_cell_v)) {
                n_cell = IntVect(AMREX_D_DECL(n_cell_v[0],n_cell_v[1],n_cell_v[2]));
            }
            pp.query("test_vector", test_vector);
            pp.query("nghost", nghost);
        }

        Box domain(IntVect(0), n_cell-1);

        Vector<iMultiFab> mf(4);
        for (int n = 0; n < mf.size(); ++n) {
            BoxArray ba(domain);
            IntVect mgs(32);
            mgs[n % AMREX_SPACEDIM] = 16;
            {
                ParmParse pp;
                int max_grid_size;
                if (pp.query("max_grid_size", max_grid_size)) {
                    mgs = IntVect{max_grid_size};
                }
            }
            ba.maxSize(mgs);
            ba.convert(IntVect::TheDimensionVector((n+1) % AMREX_SPACEDIM));
            DistributionMapping dm(ba);

            amrex::Print() << "n = " << n << ": # of boxes is " << ba.size()
                           << " the first box is " << ba[0]
                           << " nghost is " << nghost << std::endl;

            mf[n].define(ba,dm,3,nghost);
            for (MFIter mfi(mf[n]); mfi.isValid(); ++mfi) {
                Box const& bx = mfi.validbox();
                Array4<int> const& fab = mf[n].array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    fab(i,j,k,0) = i;
                    fab(i,j,k,1) = j;
                    fab(i,j,k,2) = k;
                });
            }
        }

        FillBoundary(amrex::GetVecOfPtrs(mf), Periodicity(n_cell));
        ParallelDescriptor::Barrier();
        if (test_vector) {
            BL_PROFILE("FB-vector");
            FillBoundary(amrex::GetVecOfPtrs(mf), Periodicity(n_cell));
        } else {
            BL_PROFILE("FB-old");
            for (int n = 0; n < mf.size(); ++n) {
                mf[n].FillBoundary(Periodicity(n_cell));
            }
        }

        for (int n = 0; n < mf.size(); ++n) {
            auto lendom = amrex::length(domain);
            if (AMREX_SPACEDIM == 2) lendom.z = 1;
            for (MFIter mfi(mf[n]); mfi.isValid(); ++mfi) {
                auto const& fab = mf[n].const_array(mfi);
                auto const& vbx = mfi.validbox();
                amrex::LoopOnCpu(mfi.fabbox(), [=] (int i, int j, int k)
                {
                    if (vbx.contains(Dim3{i,j,k})) {
                        if (fab(i,j,k,0) != i or fab(i,j,k,1) != j or fab(i,j,k,2) != k) {
                            amrex::AllPrint() << "Failed, valid: " << Dim3{i,j,k} << " "
                                              << fab(i,j,k,0) << " " << fab(i,j,k,1)
                                              << " " << fab(i,j,k,2) << std::endl;
                            amrex::Abort();
                        }
                    } else {
                        if (fab(i,j,k,0)%lendom.x != (i+lendom.x)%lendom.x or
                            fab(i,j,k,1)%lendom.y != (j+lendom.y)%lendom.y or
                            fab(i,j,k,2)%lendom.z != (k+lendom.z)%lendom.z) {
                            amrex::AllPrint() << "Failed, ghost: " << Dim3{i,j,k} << " " << lendom << " "
                                              << fab(i,j,k,0) << " " << fab(i,j,k,1)
                                              << " " << fab(i,j,k,2) << std::endl;
                            amrex::Abort();
                        }
                    }
                });
            }
        }
    }
    amrex::Finalize();
}
