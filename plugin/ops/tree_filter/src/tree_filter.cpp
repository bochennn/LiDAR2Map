#include <torch/extension.h>


std::tuple<at::Tensor, at::Tensor, at::Tensor>bfs_forward(
    const at::Tensor & edge_index_tensor,
    int max_adj_per_node
);

at::Tensor mst_forward(
    const at::Tensor & edge_index_tensor,
    const at::Tensor & edge_weight_tensor,
    int vertex_count
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>refine_forward(
    const at::Tensor & feature_in_tensor, 
    const at::Tensor & edge_weight_tensor, 
    const at::Tensor & sorted_index_tensor, 
    const at::Tensor & sorted_parent_index_tensor, 
    const at::Tensor & sorted_child_index_tensor 
);

at::Tensor refine_backward_feature(
    const at::Tensor & feature_in_tensor, 
    const at::Tensor & edge_weight_tensor, 
    const at::Tensor & sorted_index_tensor, 
    const at::Tensor & sorted_parent_tensor, 
    const at::Tensor & sorted_child_tensor,
    const at::Tensor & feature_out_tensor,
    const at::Tensor & feature_aggr_tensor,
    const at::Tensor & feature_aggr_up_tensor,
    const at::Tensor & weight_sum_tensor,
    const at::Tensor & weight_sum_up_tensor,
    const at::Tensor & grad_out_tensor
);

at::Tensor refine_backward_weight(
    const at::Tensor & feature_in_tensor, 
    const at::Tensor & edge_weight_tensor, 
    const at::Tensor & sorted_index_tensor, 
    const at::Tensor & sorted_parent_tensor, 
    const at::Tensor & sorted_child_tensor,
    const at::Tensor & feature_out_tensor,
    const at::Tensor & feature_aggr_tensor,
    const at::Tensor & feature_aggr_up_tensor,
    const at::Tensor & weight_sum_tensor,
    const at::Tensor & weight_sum_up_tensor,
    const at::Tensor & grad_out_tensor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mst_forward", &mst_forward, "mst forward");
    m.def("bfs_forward", &bfs_forward, "bfs forward");
    m.def("refine_forward", &refine_forward, "refine forward");
    m.def("refine_backward_feature", &refine_backward_feature, "refine backward wrt feature");
    m.def("refine_backward_weight", &refine_backward_weight, "refine backward wrt weight");
}