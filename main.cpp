
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

        IntVect n_cell{AMREX_D_DECL(64,64,64)};
        {
            Vector<int> n_cell_v;
            ParmParse pp;
            pp.queryarr("n_cell", n_cell_v);
        }

        Box domain(IntVect(0), n_cell-1);

        Vector<iMultiFab> mf(4);
        for (int n = 0; n < mf.size(); ++n) {
            BoxArray ba(domain);
            IntVect mgs(32);
            mgs[n % AMREX_SPACEDIM] = 16;
            ba.maxSize(mgs);
            ba.convert(IntVect::TheDimensionVector((n+1) % AMREX_SPACEDIM));
            DistributionMapping dm(ba);
            mf[n].define(ba,dm,3,1);
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
        {
            BL_PROFILE("FB-vector");
            FillBoundary(amrex::GetVecOfPtrs(mf), Periodicity(n_cell));
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
