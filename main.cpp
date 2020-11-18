
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
            mf[n].define(ba,dm,1,1);
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
    }
    amrex::Finalize();
}
