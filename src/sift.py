import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import os

class SIFTMatcher:
    def __init__(self, detector_params: Optional[dict] = None, matcher_type: str = 'BF'):
        default_params = {
            'nfeatures': 0,
            'nOctaveLayers': 3,
            'contrastThreshold': 0.04,
            'edgeThreshold': 10,
            'sigma': 1.6
        }
        
        if detector_params:
            default_params.update(detector_params)
        
        self.sift = cv2.SIFT_create(**default_params)

        self.matcher_type = matcher_type
        if matcher_type == 'BF':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            print("error")
        
        self.reference_image = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.query_image = None
        self.query_keypoints = None
        self.query_descriptors = None
        self.matches = None
        self.good_matches = None
    
    def load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("이미지는 파일 경로(str) 또는 numpy 배열이어야 합니다.")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def set_reference_image(self, image_input: Union[str, np.ndarray]) -> Tuple[int, int]:
        try:
            self.reference_image = self.load_image(image_input)
            
            self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(
                self.reference_image, None
            )
            
            if self.ref_descriptors is None:
                raise ValueError("참조 이미지에서 특징점을 찾을 수 없습니다.")
            
            print(f"참조 이미지 특징점 추출 완료: {len(self.ref_keypoints)}개")
            return len(self.ref_keypoints), len(self.ref_descriptors)
            
        except Exception as e:
            print(f"참조 이미지 설정 중 오류 발생: {e}")
            raise
    
    def match_with_reference(self, query_image_input: Union[str, np.ndarray], 
                           ratio_threshold: float = 0.7, 
                           min_matches: int = 10) -> Tuple[int, int]:
        if self.ref_descriptors is None:
            raise ValueError("참조 이미지가 설정되지 않았습니다. set_reference_image()를 먼저 호출하세요.")
        
        try:
            self.query_image = self.load_image(query_image_input)
            
            # 쿼리 이미지에서 특징점 추출
            self.query_keypoints, self.query_descriptors = self.sift.detectAndCompute(
                self.query_image, None
            )
            
            if self.query_descriptors is None:
                raise ValueError("쿼리 이미지에서 특징점을 찾을 수 없습니다.")
            
            # 특징점 매칭
            if self.matcher_type == 'BF':
                self.matches = self.matcher.knnMatch(
                    self.query_descriptors, self.ref_descriptors, k=2
                )
            else:  # FLANN
                self.matches = self.matcher.knnMatch(
                    self.query_descriptors, self.ref_descriptors, k=2
                )
            
            # Lowe's ratio test로 좋은 매칭 필터링
            self.good_matches = []
            for match_pair in self.matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        self.good_matches.append(m)
            
            print(f"쿼리 이미지 특징점: {len(self.query_keypoints)}개")
            print(f"전체 매칭: {len(self.matches)}개")
            print(f"좋은 매칭: {len(self.good_matches)}개")
            
            if len(self.good_matches) < min_matches:
                print(f"경고: 매칭 개수가 최소 요구사항({min_matches}개)보다 적습니다.")
            
            return len(self.matches), len(self.good_matches)
            
        except Exception as e:
            print(f"매칭 과정에서 오류 발생: {e}")
            raise
    
    def get_matches(self, return_type: str = 'keypoints') -> dict:
        if self.good_matches is None:
            raise ValueError("매칭이 수행되지 않았습니다. match_with_reference()를 먼저 호출하세요.")
        
        result = {
            'match_count': len(self.good_matches),
            'matches': self.good_matches
        }
        
        if return_type in ['keypoints', 'all']:
            result['ref_keypoints'] = self.ref_keypoints
            result['query_keypoints'] = self.query_keypoints
        
        if return_type in ['coordinates', 'all']:
            ref_points = np.float32([self.ref_keypoints[m.trainIdx].pt for m in self.good_matches])
            query_points = np.float32([self.query_keypoints[m.queryIdx].pt for m in self.good_matches])
            
            result['ref_points'] = ref_points
            result['query_points'] = query_points
        
        return result
    
    # def calculate_homography(self, ransac_threshold: float = 5.0) -> Optional[np.ndarray]:
    #     if len(self.good_matches) < 4:
    #         print("호모그래피 계산을 위해서는 최소 4개의 매칭이 필요합니다.")
    #         return None
        
    #     # 매칭된 점들의 좌표 추출
    #     ref_points = np.float32([self.ref_keypoints[m.trainIdx].pt for m in self.good_matches])
    #     query_points = np.float32([self.query_keypoints[m.queryIdx].pt for m in self.good_matches])
        
    #     # 호모그래피 계산
    #     homography, mask = cv2.findHomography(
    #         query_points, ref_points, 
    #         cv2.RANSAC, ransac_threshold
    #     )
        
    #     if homography is not None:
    #         inliers = np.sum(mask)
    #         print(f"호모그래피 계산 완료: {inliers}/{len(self.good_matches)} 인라이어")
        
    #     return homography
    
    def visualize_matches(self, max_matches: int = 50, save_path: Optional[str] = None):
        if self.good_matches is None:
            raise ValueError("매칭이 수행되지 않았습니다. match_with_reference()를 먼저 호출하세요.")
        
        display_matches = self.good_matches[:max_matches]
        
        matched_image = cv2.drawMatches(
            self.query_image, self.query_keypoints,
            self.reference_image, self.ref_keypoints,
            display_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.figure(figsize=(15, 8))
        plt.imshow(matched_image, cmap='gray')
        plt.title(f'SIFT 매칭 결과 (표시: {len(display_matches)}개 / 전체: {len(self.good_matches)}개)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"매칭 결과 저장됨: {save_path}")
        else:
            plt.show()
    
if __name__ == "__main__":
    video_path = r"c:\Users\rigel\Documents\시그마 프로젝트\repo\SIGMAxPortal301-Team2\심화과제_1\video\IMG_9860.mp4"
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        exit()
    
    matcher = SIFTMatcher()
    
    ret, first_frame = cap.read()
    if not ret:
        print("첫 번째 프레임을 읽을 수 없습니다.")
        exit()
    
    try:
        matcher.set_reference_image(first_frame)
        print("첫 번째 프레임을 참조 이미지로 설정완료")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("동영상 끝에 도달했습니다.")
                break
            
            frame_count += 1
            
            if frame_count % 10 != 0:
                continue
                
            try:
                total_matches, good_matches = matcher.match_with_reference(frame)
                
                print(f"프레임 {frame_count}: 좋은 매칭 {good_matches}개")
                
                if good_matches > 10:  
                    matched_image = cv2.drawMatches(
                        frame, matcher.query_keypoints,
                        matcher.reference_image, matcher.ref_keypoints,
                        matcher.good_matches[:20], None,  
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    height, width = matched_image.shape[:2]
                    scale = 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized = cv2.resize(matched_image, (new_width, new_height))
                    
                    cv2.imshow('SIFT Matching', resized)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                        
            except Exception as e:
                print(f"프레임 {frame_count} 처리 중 오류: {e}")
                continue
    
    except Exception as e:
        print(f"에러: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("동영상 처리 완료")