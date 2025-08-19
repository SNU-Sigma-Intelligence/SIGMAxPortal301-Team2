#include <math.h>

// 3x3 행렬 출력용 (디버깅 시 시리얼 모니터 확인)
void printMatrix(const float R[3][3]) {
    for (int i = 0; i < 3; i++) {
        Serial.print("[ ");
        for (int j = 0; j < 3; j++) {
            Serial.print(R[i][j], 6);
            Serial.print(" ");
        }
        Serial.println("]");
    }
}

// 벡터 정규화
void normalize(float v[3]) {
    float norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm > 1e-6) {
        v[0] /= norm; v[1] /= norm; v[2] /= norm;
    }
}

// 두 벡터의 외적
void crossProduct(const float a[3], const float b[3], float result[3]) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}

// 두 벡터의 내적
float dotProduct(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// 회전행렬 계산 
// ref[3] : 기준 행렬
// target[3] : 대상 행렬
// ref * R = target
void computeRotation(const float ref[3], const float target[3], float R[3][3]) {
    float u[3] = {ref[0], ref[1], ref[2]};
    float v[3] = {target[0], target[1], target[2]};
    normalize(u);
    normalize(v);

    float axis[3];
    
    crossProduct(u, v, axis);
    float sinTheta = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    float cosTheta = dotProduct(u, v);

    if (sinTheta > 1e-6) {
        axis[0] /= sinTheta;
        axis[1] /= sinTheta;
        axis[2] /= sinTheta;
    }

    float x = axis[0], y = axis[1], z = axis[2];

    // 로드리게스 공식 사용
    R[0][0] = cosTheta + x*x*(1-cosTheta);
    R[0][1] = x*y*(1-cosTheta) - z*sinTheta;
    R[0][2] = x*z*(1-cosTheta) + y*sinTheta;

    R[1][0] = y*x*(1-cosTheta) + z*sinTheta;
    R[1][1] = cosTheta + y*y*(1-cosTheta);
    R[1][2] = y*z*(1-cosTheta) - x*sinTheta;

    R[2][0] = z*x*(1-cosTheta) - y*sinTheta;
    R[2][1] = z*y*(1-cosTheta) + x*sinTheta;
    R[2][2] = cosTheta + z*z*(1-cosTheta);
}

// 메인 함수: 센서 3개 데이터를 입력받아 R_12, R_23, R_31 계산
void computeAllRotations(float gyroData[3][3]) {
    float R_12[3][3], R_23[3][3], R_31[3][3];

    computeRotation(gyroData[0], gyroData[1], R_12);
    computeRotation(gyroData[1], gyroData[2], R_23);
    computeRotation(gyroData[2], gyroData[0], R_31);

    Serial.println("R_12:");
    printMatrix(R_12);
    Serial.println("R_23:");
    printMatrix(R_23);
    Serial.println("R_31:");
    printMatrix(R_31);
}

void setup() {
    Serial.begin(115200);

    // 예제: 각가속도 벡터 (센서 1,2,3)
    float gyroData[3][3] = {
        {0.1, 0.5, 0.3},   // 센서1
        {0.2, 0.4, 0.35},  // 센서2
        {0.15, 0.55, 0.25} // 센서3
    };

    computeAllRotations(gyroData);
}

void loop() {
    // 필요시 반복 측정
}
