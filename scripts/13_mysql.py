"""
13 — Xoá các sản phẩm không thuộc results/unique_product_ids.txt khỏi MySQL.
Các bảng xoá theo thứ tự (lá → gốc) để tránh lỗi foreign key constraint.

Cách dùng:
   python scripts/13_mysql.py

Yêu cầu:
   - Đã chạy scripts/14_get_unique_product_ids.py (có results/unique_product_ids.txt)
   - Kết nối MySQL thành công (cấu hình bên dưới)

Output:
   In thống kê số lượng bản ghi đã xoá ở mỗi bảng
"""
import os
import sys
import mysql.connector
from mysql.connector import Error

# Thêm project root vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULT_DIR


def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            port=3306,
            database="food",
            user="giang",
            password="123456789",
        )
        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✅ Đã kết nối thành công tới MySQL Server phiên bản: {db_info}")
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print(f"✅ Bạn đang kết nối tới cơ sở dữ liệu: {record[0]}")
            return connection
    except Error as e:
        print(f"❌ Lỗi khi kết nối tới MySQL: {e}")
        return None


def load_keep_ids():
    """Đọc danh sách product_id cần GIỮ LẠI từ results/unique_product_ids.txt."""
    path = os.path.join(RESULT_DIR, "unique_product_ids.txt")
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
        print(f"   Hãy chạy scripts/14_get_unique_product_ids.py trước.")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f if line.strip()]

    print(f"📄 Đã đọc {len(ids)} product_id cần giữ lại từ {path}")
    return ids


def format_id_tuple(id_list):
    """Format list string IDs thành SQL tuple an toàn (các id là số nguyên)."""
    if not id_list:
        return "()"
    # Kiểm tra tất cả là số
    nums = [int(x) for x in id_list]
    return "(" + ", ".join(str(x) for x in nums) + ")"


def delete_products_not_in_list(conn, keep_ids):
    """
    Xoá các sản phẩm có id KHÔNG nằm trong keep_ids.
    Xoá theo thứ tự: bảng con → bảng cha.
    """
    cursor = conn.cursor()
    keep_tuple = format_id_tuple(keep_ids)

    # ---- Bước 0: Thống kê trước khi xoá ----
    print("\n📊 THỐNG KÊ TRƯỚC KHI XOÁ")
    cursor.execute("SELECT COUNT(*) FROM products")
    total_products = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM products WHERE id NOT IN {keep_tuple}")
    to_delete_products = cursor.fetchone()[0]
    print(f"   Tổng số products:          {total_products:>8,}")
    print(f"   Số products cần GIỮ:      {total_products - to_delete_products:>8,}")
    print(f"   Số products cần XOÁ:      {to_delete_products:>8,}")
    print()

    if to_delete_products == 0:
        print("✅ Không có sản phẩm nào cần xoá.")
        return

    # ---- Bước 1: Xoá các bảng con của product_variants ----
    # Những bảng này có variant_id (hoặc các tên khác) tham chiếu product_variants.id

    # product_variant_review_media → product_variant_review
    print("🗑️  Xoá product_variant_review_media...")
    cursor.execute(f"""
        DELETE prm FROM product_variant_review_media prm
        JOIN product_variant_review prv ON prm.review_id = prv.id
        JOIN product_variants pv ON prv.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # product_variant_review
    print("🗑️  Xoá product_variant_review...")
    cursor.execute(f"""
        DELETE prv FROM product_variant_review prv
        JOIN product_variants pv ON prv.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # cart_gift_items
    print("🗑️  Xoá cart_gift_items...")
    cursor.execute(f"""
        DELETE cgi FROM cart_gift_items cgi
        JOIN product_variants pv ON cgi.product_variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # cart_combo_items
    print("🗑️  Xoá cart_combo_items...")
    cursor.execute(f"""
        DELETE cci FROM cart_combo_items cci
        JOIN product_variants pv ON cci.product_variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # cart_items
    print("🗑️  Xoá cart_items...")
    cursor.execute(f"""
        DELETE ci FROM cart_items ci
        JOIN product_variants pv ON ci.product_variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # discount_buy_n_get_m_items (buy_variant_id, free_variant_id)
    print("🗑️  Xoá discount_buy_n_get_m_items...")
    cursor.execute(f"""
        DELETE dbi FROM discount_buy_n_get_m_items dbi
        JOIN product_variants pv ON dbi.buy_variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted_buy = cursor.rowcount
    cursor.execute(f"""
        DELETE dbi FROM discount_buy_n_get_m_items dbi
        JOIN product_variants pv ON dbi.free_variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted_free = cursor.rowcount
    print(f"   → {deleted_buy + deleted_free} bản ghi đã xoá")

    # discount_combo_gift_items
    print("🗑️  Xoá discount_combo_gift_items...")
    cursor.execute(f"""
        DELETE dcgi FROM discount_combo_gift_items dcgi
        JOIN product_variants pv ON dcgi.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # discount_combo_items
    print("🗑️  Xoá discount_combo_items...")
    cursor.execute(f"""
        DELETE dci FROM discount_combo_items dci
        JOIN product_variants pv ON dci.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # discount_invoice_gift_items
    print("🗑️  Xoá discount_invoice_gift_items...")
    cursor.execute(f"""
        DELETE digi FROM discount_invoice_gift_items digi
        JOIN product_variants pv ON digi.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # discount_invoice_variants
    print("🗑️  Xoá discount_invoice_variants...")
    cursor.execute(f"""
        DELETE div FROM discount_invoice_variants div
        JOIN product_variants pv ON div.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # discount_percentage_variants
    print("🗑️  Xoá discount_percentage_variants...")
    cursor.execute(f"""
        DELETE dpv FROM discount_percentage_variants dpv
        JOIN product_variants pv ON dpv.variant_id = pv.id
        WHERE pv.product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # ---- Bước 2: Xoá product_variants ----
    print("🗑️  Xoá product_variants...")
    cursor.execute(f"""
        DELETE FROM product_variants
        WHERE product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # ---- Bước 3: Xoá product_media ----
    print("🗑️  Xoá product_media...")
    cursor.execute(f"""
        DELETE FROM product_media
        WHERE product_id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # ---- Bước 4: Xoá products ----
    print("🗑️  Xoá products...")
    cursor.execute(f"""
        DELETE FROM products
        WHERE id NOT IN {keep_tuple}
    """)
    deleted = cursor.rowcount
    print(f"   → {deleted} bản ghi đã xoá")

    # ---- Commit ----
    conn.commit()
    print("\n✅ Đã commit tất cả thay đổi thành công!")


def verify_result(conn, keep_ids):
    """Kiểm tra sau khi xoá: chỉ còn các sản phẩm trong keep_ids."""
    cursor = conn.cursor()
    keep_tuple = format_id_tuple(keep_ids)

    cursor.execute("SELECT COUNT(*) FROM products")
    remaining = cursor.fetchone()[0]
    print(f"\n📊 KIỂM TRA SAU KHI XOÁ")
    print(f"   Số products còn lại: {remaining:,}")

    # Kiểm tra xem còn product nào không nằm trong keep_ids không
    cursor.execute(f"""
        SELECT COUNT(*) FROM products
        WHERE id NOT IN {keep_tuple}
    """)
    orphan = cursor.fetchone()[0]
    if orphan > 0:
        print(f"   ⚠️  Vẫn còn {orphan} product không thuộc danh sách giữ lại!")
    else:
        print(f"   ✅ Tất cả products còn lại đều thuộc danh sách giữ lại.")

    # Kiểm tra variant còn dư
    cursor.execute(f"""
        SELECT COUNT(*) FROM product_variants pv
        JOIN products p ON pv.product_id = p.id
        WHERE p.id NOT IN {keep_tuple}
    """)
    orphan_var = cursor.fetchone()[0]
    print(f"   Số variant còn lại: {orphan_var:,} (không thuộc products giữ lại)")

    # Kiểm tra bảng con của variant
    cursor.execute(f"""
        SELECT COUNT(*) FROM cart_items ci
        JOIN product_variants pv ON ci.product_variant_id = pv.id
        JOIN products p ON pv.product_id = p.id
        WHERE p.id NOT IN {keep_tuple}
    """)
    orphan_cart = cursor.fetchone()[0]
    print(f"   Số cart_items còn dư: {orphan_cart:,}")


def main():
    print("=" * 60)
    print("  SCRIPT 13: XOÁ SẢN PHẨM KHÔNG THUỘC UNIQUE_PRODUCT_IDS.TXT")
    print("=" * 60)

    # 1. Kết nối MySQL
    conn = connect_to_mysql()
    if not conn:
        return

    try:
        # 2. Đọc danh sách ID cần giữ
        keep_ids = load_keep_ids()

        if len(keep_ids) == 0:
            print("⚠️  Danh sách keep_ids rỗng. Hủy thao tác.")
            return

        # 3. Xoá
        delete_products_not_in_list(conn, keep_ids)

        # 4. Kiểm tra
        verify_result(conn, keep_ids)

    except Error as e:
        print(f"❌ Lỗi MySQL: {e}")
        conn.rollback()
        print("⚠️  Đã rollback do lỗi.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        conn.rollback()
        print("⚠️  Đã rollback do lỗi.")
    finally:
        if conn and conn.is_connected():
            conn.close()
            print("🔌 Đã đóng kết nối an toàn.")


if __name__ == "__main__":
    main()