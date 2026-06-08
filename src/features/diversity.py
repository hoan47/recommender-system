"""
Diversity filter — đảm bảo danh sách gợi ý có nhiều department khác nhau

Dựa trên model 1 (RS_Instacart-main/models.py dòng 291-315)

Thuật toán: Greedy diversity selection
  1. Duyệt ứng viên theo score giảm dần
  2. Đếm số sản phẩm đã chọn theo dept
  3. Nếu dept này đã đạt MAX_PER_DEPT sản phẩm → bỏ qua (trừ khi cùng dept với seed)
  4. Nếu cùng dept với seed và đã đạt SAME_DEPT_MAX → bỏ qua

Tham số:
    MAX_PER_DEPT = 5   # Tối đa SP mỗi department (khác dept seed)
    SAME_DEPT_MAX = 3  # Tối đa SP cùng department với seed
"""

from collections import defaultdict

# Default thresholds
MAX_PER_DEPT = 5
SAME_DEPT_MAX = 3


def _get_dept(pid: int, prod_dept_map: dict) -> int:
    """Lấy department_id của sản phẩm, trả về -1 nếu không tìm thấy."""
    return prod_dept_map.get(pid, -1)


def diversity_filter(
    seed_pid: int,
    ranked_pids: list,
    prod_dept_map: dict,
    k: int = 100,
    max_per_dept: int = MAX_PER_DEPT,
    same_dept_max: int = SAME_DEPT_MAX
) -> list:
    """
    Lọc danh sách gợi ý để đảm bảo đa dạng department.

    Args:
        seed_pid: int — sản phẩm gốc (seed)
        ranked_pids: list[int] — danh sách ứng viên đã xếp hạng theo score giảm dần
        prod_dept_map: dict[int, int] — mapping product_id → department_id
        k: int — số lượng kết quả mong muốn
        max_per_dept: int — tối đa SP cùng dept (khác dept seed)
        same_dept_max: int — tối đa SP cùng dept với seed

    Returns:
        list[int] — danh sách đã lọc, tối đa k sản phẩm
    """
    seed_dept = _get_dept(seed_pid, prod_dept_map)
    dept_count = defaultdict(int)
    result = []

    for pid in ranked_pids:
        if len(result) >= k:
            break

        d = _get_dept(pid, prod_dept_map)
        if d == seed_dept:
            # Cùng dept với seed: giới hạn số lượng
            if dept_count[d] < same_dept_max:
                dept_count[d] += 1
                result.append(pid)
        else:
            # Khác dept: giới hạn số lượng
            if dept_count[d] < max_per_dept:
                dept_count[d] += 1
                result.append(pid)

    return result