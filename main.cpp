#include "hessian.h"

int main(){
    size_t N = 15;
    size_t nghost = 1;
    size_t N_ext = N+nghost;
    typename Field<double>::vector3d_t h{1e-1, 1e-1, 1e-1};

	// Scalar Fields
    Field<double> field(N+nghost, h);
    Field<double> field_result(N+nghost, h);
    Field<double> field_exact(N+nghost, h);

	// Matrix Fields
	Field<double> gaussian_field(N+nghost, h);
	MatrixField mfield(N+nghost, h);
	MatrixField mfield_result(N+nghost, h);
	MatrixField mfield_exact(N+nghost, h);


    // Initialize
    field.init_x2y2z2();
    field_exact.init_x2y2z2_deriv();
    gaussian_field.init_gaussian();
	mfield_exact.initGaussHess();


	////////////////////////////////////////////
	// Test individual Differential Operators //
	////////////////////////////////////////////

    // Construct differential operator
    DiffOpChain<Dim::X, double, DiffType::Forward, 
                DiffOpChain<Dim::Z, double, DiffType::Backward,
                            Field<double>>> xzDiff(field);

    double error = 0.0;
    for(size_t i = 2*nghost; i < N_ext-2*nghost; ++i){
        for(size_t j = 2*nghost; j < N_ext-2*nghost; ++j){
            for(size_t k = 2*nghost; k < N_ext-2*nghost; ++k){
                double approx_value = xzDiff(Index({i,j,k},N_ext));
                field_result(i,j,k) = approx_value;

                // Compute relative error
                error += relative_error(field_exact(i,j,k), approx_value);
            }
        }
    }

    std::cout << "Relative Error chained differential operators: " << std::setw(10) << error << std::endl;


	/////////////////////////////////////
	// Test Hessian Matrix computation //
	/////////////////////////////////////

	// Construct Hessian operator
    GeneralizedHessOp<double,DiffType::Forward,DiffType::Forward,DiffType::Forward> hessOp(field);

    double hess_error = 0.0;
    for(size_t i = 2*nghost; i < N_ext-2*nghost; ++i){
        for(size_t j = 2*nghost; j < N_ext-2*nghost; ++j){
            for(size_t k = 2*nghost; k < N_ext-2*nghost; ++k){

                MatrixField::matrix3d_t approx_hess = hessOp(Index({i,j,k}, N_ext));
                mfield_result(i,j,k) = approx_hess;
                for(size_t dim0 = 0; dim0 < 3; ++dim0){
                    for(size_t dim1 = 0; dim1 < 3; ++dim1){
                        approx_hess(dim0, dim1);

                        // Compute relative error
                        hess_error += relative_error(mfield_exact(i,j,k)(dim0,dim1), approx_hess(dim0, dim1));
                    }
                }
            }
        }
    }

    std::cout << "Relative Error Hessian Operator: " << std::setw(10) << hess_error << std::endl;
	

    return 0;
}
