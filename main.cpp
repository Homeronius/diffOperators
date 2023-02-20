#include "hessian.h"

int main(){
    using namespace Mesh;
    size_t nghost = 1;
    vector3d_t h{0.5e-1, 0.5e-1, 0.5e-1};

	// Scalar Fields
    Field<double> field(h);
    Field<double> field_result(h);
    Field<double> field_exact(h);

    vector3i_t N = field.getN();
    vector3i_t N_ext = field.getN_ext();

    // Initialize
    field.init_x2y2z2();
    field_exact.init_x2y2z2_deriv();


	////////////////////////////////////////////
	// Test individual Differential Operators //
	////////////////////////////////////////////

    // Construct differential operator
    DiffOpChain<Dim::X, double, DiffType::Centered, 
                DiffOpChain<Dim::Z, double, DiffType::Centered,
                            Field<double>>> xzDiff(field);
    //DiffOpChain<Dim::X, double, DiffType::CenteredDeriv2, 
                            //Field<double>> xzDiff(field);

    double error = 0.0;
    for(size_t i = 2*nghost; i < N_ext[0]-2*nghost; ++i){
        for(size_t j = 2*nghost; j < N_ext[1]-2*nghost; ++j){
            for(size_t k = 2*nghost; k < N_ext[2]-2*nghost; ++k){
                double approx_value = xzDiff(Index({i,j,k},N_ext));
                field_result(i,j,k) = approx_value;

                // Compute L2 error
                error += pow(field_exact(i,j,k) - approx_value, 2);
            }
        }
    }

    std::cout << "Absolute L2 Error of chained differential operators: " << std::setw(10) << sqrt(error) << std::endl;


	/////////////////////////////////////
	// Test Hessian Matrix computation //
	/////////////////////////////////////
    
	// Matrix Fields
	Field<matrix3d_t> mfield(h);
	Field<matrix3d_t> mfield_result(h);
	Field<matrix3d_t> mfield_exact(h);

    N = field.getN();
    N_ext = field.getN_ext();
    
    // Initialize
    field.init_xyz();
    bool gaussian_ic = 0;
	mfield_exact.initHess(gaussian_ic);
    
	// Construct Hessian operator
    GeneralizedHessOp<double,DiffType::Centered,DiffType::Centered,DiffType::Centered> hessOp(field);
    //GeneralizedHessOp<double,DiffType::Forward,DiffType::Forward,DiffType::Forward> hessOp(field);
    //GeneralizedHessOp<double,DiffType::Backward,DiffType::Backward,DiffType::Backward> hessOp(field);
 
    DiffOpChain<Dim::X, double, DiffType::Centered, 
        DiffOpChain<Dim::X, double, DiffType::Centered,
    //DiffOpChain<Dim::X, double, DiffType::Forward,
        //DiffOpChain<Dim::X, double, DiffType::Forward,
    //DiffOpChain<Dim::X, double, DiffType::Backward,
        //DiffOpChain<Dim::X, double, DiffType::Backward,
        Field<double>>> xxDiff(field);

    matrix3d_t hess_error;
    hess_error.setZero();
    double avg_hess_error = 0.0;
    for(size_t i = 4*nghost; i < N_ext[0]-4*nghost; ++i){
        for(size_t j = 4*nghost; j < N_ext[1]-4*nghost; ++j){
            for(size_t k = 4*nghost; k < N_ext[2]-4*nghost; ++k){

                matrix3d_t approx_hess = hessOp(Index({i,j,k}, N_ext));
                mfield_result(i,j,k) = approx_hess;
                if (i == 4 && j == 5 && k == 10){
                    std::cout << "Hessian Approximation\n" << approx_hess << std::endl;
                    std::cout << "\nHessian Ground truth\n" << mfield_exact(i,j,k) << std::endl;
                    std::cout << "\nd2/dx2 mfield_exact(0,0) = " << xxDiff(Index({i,j,k},N_ext)) << std::endl;
                    double hxInv = field.getInvMeshSpacing()[0];
                    std::cout << "\nd2/dx2 manual(0,0) = " << hxInv*hxInv*(field(i+1,j,k) - 2.0*field(i,j,k) + field(i-1,j,k)) << std::endl;
                }
                for(size_t dim0 = 0; dim0 < 3; ++dim0){
                    for(size_t dim1 = 0; dim1 < 3; ++dim1){
                        hess_error(dim0, dim1) += pow(mfield_exact(i,j,k)(dim0,dim1) - approx_hess(dim0, dim1), 2);
                    }
                }
            }
        }
    }


    std::cout << "=================================================" << std::endl;
    std::cout << "Error Hessian Operator:\n" << hess_error.cwiseSqrt() << std::endl;
    std::cout << "\nAverage Error Hessian Operator: " << hess_error.sum()/hess_error.size() << std::endl;
	

    return 0;
}
