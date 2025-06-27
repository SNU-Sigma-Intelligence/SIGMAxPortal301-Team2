# SIGMAxPortal301-Team2

## 연습과제
T1. Marker-based Tracking
Pnp를 통해 마커가 있는 물체(UIR LED plate)의 자세를 추정합니다.
## 심화과제
1. Random Marker Problem
T1에서는 plate 상의 마커 위치가 정확히 배치되어있었습니다. 임의의 오브젝트에 임의로 마커를 붙인 상황에서 마커들간의 위치관계를 추정하고, 최종적으로 초깃값 대비 실시간 오브젝트 자세를 추정해봅니다.
2. Multi-camera Tracking System
사람이 움직이다보면 물체가 가려질 수도 있겠죠? 카메라를 이용할 때의 최대 단점은 occlusion 문제입니다. 2대 이상의 카메라를 이용하여 occlusion 문제를 개선해봅니다.
3. Marker+IMU Sensor Fusiuon
2D 카메라를 통해 측정된 값들은 시간에 따른 누적오차(low-frequency noise)가 발생하지 않지만 측정치 자체가 불안정(high-frequency noise)할 수 있습니다. 반면 관성센서는 측정치는 안정적이지만 시간에 따른 누적오차가 발생한다는 문제가 있습니다. 본 과제에서는 Marker-based Tracker에 6축관성센서를 결합하여 서로의 단점을 상쇄시킴으로서 데이터의 정확도를 높이는 것을 목표로 합니다.
