#include "hessian.h"

int main(){
    size_t N = 20;
    size_t nghost = 1;
    size_t N_ext = N+nghost;
    std::array<double, 3> h{1e-1, 1e-1, 1e-1};
    Field<double> field(N+nghost, h, 1.0);
    Field<double> field_result(N+nghost, h);

    // Construct differential operator
    DiffOpBase<Dim::X, double, DiffType::Backward> xDiff(field);
    field.print(5);

    for(size_t i = nghost; i < N_ext-nghost; ++i){
        for(size_t j = nghost; j < N_ext-nghost; ++j){
            for(size_t k = nghost; k < N_ext-nghost; ++k){
                field_result(i,j,k) = xDiff(i,j,k);
            }
        }
    }

    field_result.print(10);

    return 0;
}
