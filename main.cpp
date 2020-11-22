
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box domain(IntVect(0), IntVect(63));

        Vector<FabArray<IArrayBox> > mf(4);
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

        FillBoundary(amrex::GetVecOfPtrs(mf));

        for (int n = 0; n < mf.size(); ++n) {
            for (MFIter mfi(mf[n]); mfi.isValid(); ++mfi) {
                auto const& fab = mf[n].const_array(mfi);
                amrex::LoopOnCpu(mfi.fabbox(), [=] (int i, int j, int k)
                {
                    if (domain.contains(Dim3{i,j,k})) {
                        if (fab(i,j,k,0) != i or fab(i,j,k,1) != j or fab(i,j,k,2) != k) {
                            amrex::AllPrint() << "Failed: " << Dim3{i,j,k} << " "
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
