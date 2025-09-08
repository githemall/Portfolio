import os

# -------------------------------------------------------------
# ❗️ 사용 전 설정할 부분 ❗️
# 1. 파일이 들어있는 폴더 경로를 지정하세요.
#    - 예시: "C:/Users/MyUser/Desktop/ramen_dataset/train/sinramen"
#    - 팁: 폴더를 열고 주소창을 복사하면 편해요. (백슬래시'\'는 슬래시'/'로 바꿔주세요)
target_folder = "/home/asd/projects/Portfolio/My_image_classification/rabbit"

# 2. 원하는 파일 이름의 앞부분(prefix)을 정하세요.
prefix = "rabbit"
# -------------------------------------------------------------


def rename_files(folder_path, name_prefix):
    """
    지정된 폴더 안의 파일 이름을 일괄적으로 변경합니다.
    (예: a.jpg, b.png -> sinramen_001.jpg, sinramen_002.png)
    """
    
    # 1. 폴더 안의 파일 목록을 불러옵니다.
    try:
        file_list = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"⚠️ 에러: '{folder_path}' 폴더를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    # 2. 파일명을 하나씩 바꾸는 작업을 반복합니다.
    count = 1
    for filename in file_list:
        # 파일의 전체 경로를 만듭니다.
        old_path = os.path.join(folder_path, filename)

        # 폴더가 아닌 파일인 경우에만 이름을 변경합니다.
        if os.path.isfile(old_path):
            # 파일의 확장자를 분리합니다. (예: .jpg, .png)
            name, ext = os.path.splitext(filename)
            
            # 새로운 파일 이름을 형식에 맞게 만듭니다. (예: sinramen_001.jpg)
            # str(count).zfill(3)는 숫자를 3자리로 만들고 빈자리는 0으로 채워줍니다. (1 -> 001)
            new_name = f"{name_prefix}_{str(count).zfill(3)}{ext}"
            new_path = os.path.join(folder_path, new_name)
            
            # 파일 이름을 변경합니다.
            os.rename(old_path, new_path)
            
            count += 1
            
    print(f"✅ 총 {count - 1}개의 파일 이름이 성공적으로 변경되었습니다.")
    print(f"👉 변경된 폴더: {folder_path}")


# 스크립트 실행
if __name__ == "__main__":
    if target_folder == "여기에 폴더 경로를 입력하세요" or prefix == "":
        print("❗️ 설정이 필요합니다. target_folder와 prefix 변수를 수정해주세요.")
    else:
        rename_files(target_folder, prefix)