#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

template <typename BaseT, typename QueryT>
concept ValidSegTreeTypes = requires(BaseT b, QueryT q) {
    // Can default construct query type with base type
    std::is_constructible_v<QueryT, BaseT>;

    // can merge queries
    { q + q };
};

template <typename BaseT, typename QueryT, bool LazyPropogation = true>
    requires std::is_arithmetic_v<BaseT> && ValidSegTreeTypes<BaseT, QueryT>
class SegmentTree {
    std::size_t size;
    std::vector<std::optional<QueryT>> data;
    using UpdaterT = QueryT(QueryT, std::size_t);
    using RangeUpdaterT = QueryT(QueryT, std::size_t, std::size_t);
    std::vector<std::vector<std::variant<
        std::monostate, std::function<UpdaterT>, std::function<RangeUpdaterT>>>>
        lazyTasks;

    constexpr std::size_t _get_segtree_size() const noexcept {
        return 4 * size + 1;
    }

    constexpr void _merge_childs(std::size_t node) noexcept {
        assert(2 * node + 1 < data.size());
        data[node] = data[2 * node].value() + data[2 * node + 1].value();
    }

    constexpr void _build_tree(std::size_t node, std::size_t tree_left,
                     std::size_t tree_right,
                     const std::vector<BaseT>& input_data) noexcept {
        if (tree_left != tree_right) {
            std::size_t mid = (tree_left + tree_right) >> 1;
            _build_tree(2 * node, tree_left, mid, input_data);
            _build_tree(2 * node + 1, mid + 1, tree_right, input_data);
            _merge_childs(node);
        } else {
            data[node] = QueryT(input_data[tree_left]);
        }
    }

    template <typename Updater>
    constexpr void _update_rebuild_tree(Updater&& updater, std::size_t node,
                              std::size_t tree_left, std::size_t tree_right) noexcept {
        if (std::is_same_v<UpdaterT, Updater>) {
            data[node] = updater(data[node], tree_left);
        } else if (std::is_same_v<RangeUpdaterT, UpdaterT>) {
            data[node] = updater(data[node], tree_left, tree_right);
        } else {
            static_assert(false, "Invalid updater type");
        }
        if (tree_left != tree_right) {
            std::size_t mid = (tree_left + tree_right) >> 1;
            _update_rebuild_tree(updater, 2 * node, tree_left, mid);
            _update_rebuild_tree(updater, 2 * node + 1, mid + 1, tree_right);
            _merge_childs(node);
        }
    }

    constexpr std::optional<QueryT> _query_tree(std::size_t node, std::size_t tree_left,
                                      std::size_t tree_right,
                                      std::size_t query_left,
                                      std::size_t query_right) noexcept {
        if constexpr (LazyPropogation) {
            _execute_lazy_tasks_and_pushdown(node, tree_left, tree_right);
        }
        if (tree_left > query_right || tree_right < query_left) {
            return std::nullopt;
        }
        if (query_left <= tree_left && tree_right <= query_right) {
            return data[node];
        }
        std::size_t mid = (tree_left + tree_right) >> 1;
        auto left_query =
            _query_tree(2 * node, tree_left, mid, query_left, query_right);
        auto right_query = _query_tree(2 * node + 1, mid + 1, tree_right,
                                       query_left, query_right);
        if (left_query && right_query) {
            return left_query.value() + right_query.value();
        } else if (left_query) {
            return left_query;
        } else {
            return right_query;
        }
        return std::nullopt;
    }

    template <typename Updater>
    constexpr void _update_tree(Updater&& updater, std::size_t node,
                      std::size_t tree_left, std::size_t tree_right,
                      std::size_t update_left, std::size_t update_right) noexcept {
        if constexpr (LazyPropogation) {
            _execute_lazy_tasks_and_pushdown(node, tree_left, tree_right);
        }
        if (tree_left > update_right || tree_right < update_left) {
            return;
        }
        if (update_left <= tree_left && tree_right <= update_right) {
            if constexpr (LazyPropogation) {
                lazyTasks[node].emplace_back(updater);
            } else {
                _update_rebuild_tree(updater, node, tree_left, tree_right);
            }
            return;
        }
        std::size_t mid = (tree_left + tree_right) >> 1;
        _update_tree(updater, 2 * node, tree_left, mid, update_left,
                     update_right);
        _update_tree(updater, 2 * node + 1, mid + 1, tree_right, update_left,
                     update_right);
        _merge_childs(node);
    }

    constexpr void _execute_lazy_tasks_and_pushdown(std::size_t node,
                                          std::size_t tree_left,
                                          std::size_t tree_right) noexcept
        requires LazyPropogation
    {
        for (auto&& task : lazyTasks[node]) {
            if (std::holds_alternative<std::function<UpdaterT>>(task)) {
                data[node] =
                    std::get<std::function<UpdaterT>>(task)(data[node].value(), tree_left);
                lazyTasks[2 * node].emplace_back(task);
                lazyTasks[2 * node + 1].emplace_back(task);
            } else if (std::holds_alternative<std::function<RangeUpdaterT>>(task)) {
                data[node] = std::get<std::function<RangeUpdaterT>>(task)(
                    data[node].value(), tree_left, tree_right);
                lazyTasks[2 * node].emplace_back(task);
                lazyTasks[2 * node + 1].emplace_back(
                    task);
            }
        }
        lazyTasks[node].clear();
    }

   public:
    constexpr SegmentTree(std::size_t size) : size(size) {
        data.resize(_get_segtree_size());
        if constexpr (LazyPropogation) {
            lazyTasks.resize(_get_segtree_size());
        }
        _build_tree(1, 0, size - 1, std::vector(size, BaseT{}));
    }

    constexpr SegmentTree(std::size_t size, BaseT default_value) : size(size) {
        data.resize(_get_segtree_size());
        if constexpr (LazyPropogation) {
            lazyTasks.resize(_get_segtree_size());
        }
        _build_tree(1, 0, size - 1, std::vector(size, default_value));
    }

    constexpr SegmentTree(const std::vector<BaseT>& input_data)
        : size(input_data.size()) {
        data.resize(_get_segtree_size());
        if constexpr (LazyPropogation) {
            lazyTasks.resize(_get_segtree_size());
        }
        _build_tree(1, 0, size - 1, input_data);
    }

    constexpr std::optional<QueryT> query(std::size_t position) noexcept {
        assert(0 <= position && position < size);
        return _query_tree(1, 0, size - 1, position, position);
    }

    constexpr std::optional<QueryT> query(std::size_t left, std::size_t right) noexcept {
        assert(0 <= left && left <= right && right < size);
        return _query_tree(1, 0, size - 1, left, right);
    }

    template <typename UpdaterT>
    constexpr void update(UpdaterT&& updater, std::size_t position) noexcept {
        assert(0 <= position && position < size);
        _update_tree(updater, 1, 0, size - 1, position, position);
    }

    template <typename RangeUpdaterT>
    constexpr void update(RangeUpdaterT&& updater, std::size_t left, std::size_t right) noexcept {
        assert(0 <= left && left <= right && right < size);
        _update_tree(updater, 1, 0, size - 1, left, right);
    }
};
