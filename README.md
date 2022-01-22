# Đồ Án Cuối Kì Môn Lập trình song song trên GPU.
## Thành viên nhóm:
- Nguyễn Đình Hoàng Phúc - MSSV:18120143.
- Lê Minh Khoa - MSSV: 18120415.

## Cách biên dịch chương trình:
```
!mkdir build
%cd build
!cmake ..
!make
```
**Note**: Nếu GPU là K80 (có compute capability là 3.7) thì cần phải uncomment dòng cuối trong file CMakeLists.txt để biên dịch ra đúng.

## Cách thực thi chương trình:
`./FinalProject <input path> <output path> <solutionID> <n seam> (<blockSize.x> <blockSize.y>)`

Với: 
- `input path`: Đường dẫn đến file ảnh input dạng pnm.
- `output path`: Đường dẫn đến file ảnh input dạng pnm.
- `solutionID`: Số thứ tự của phiên bản cài đặt song song muốn chạy là một con số từ 0-7. 
- `n seam`: Số đường seam cần xóa (kích thước cần giảm theo chiều ngang của ảnh).
- `blockSize.x` và `blockSize.y`: block size cho cài đặt song song sử dụng.

Chương trình sau khi thực thi sẽ cho ra:
- File `sequential_solution.pnm`: ảnh kết quả của cài đặt tuần tự.
- File có tên lấy từ tham số `output path`: ảnh kết quả của cài đặt song song có số thứ từ là `solutionID`.
- In ra màn hình thời gian chạy của các cài đặt và so sánh lỗi giữa cài đặt tuần tự và song song.

Lưu ý: nếu `solutionID = 0`, chương trình sẽ thực thi toàn bộ mọi phiên bản cài đặt song song.
