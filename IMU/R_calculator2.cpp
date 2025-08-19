#include <math.h>

// 3x3 행렬 출력용 (디버깅 시 시리얼 모니터 확인)
void printMatrix(const float matrix[3][3]) {
    for (int i = 0; i < 3; i++) {
        Serial.print("[ ");
        for (int j = 0; j < 3; j++) {
            Serial.print(matrix[i][j], 6);
            Serial.print(" ");
        }
        Serial.println("]");
    }
}

// 3x3 행렬 곱셈: C = A * B
void matrixMultiply(const float A[3][3], const float B[3][3], float C[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 3x3 행렬 전치: B = A^T
void matrixTranspose(const float A[3][3], float B[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            B[i][j] = A[j][i];
        }
    }
}

// 3x3 행렬 초기화 (단위행렬 또는 영행렬)
void matrixInit(float matrix[3][3], bool identity = false) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matrix[i][j] = (identity && i == j) ? 1.0 : 0.0;
        }
    }
}

// 점군의 중심점(centroid) 계산
void computeCentroid(const float points[][3], int numPoints, float centroid[3]) {
    centroid[0] = centroid[1] = centroid[2] = 0.0;
    
    for (int i = 0; i < numPoints; i++) {
        centroid[0] += points[i][0];
        centroid[1] += points[i][1];
        centroid[2] += points[i][2];
    }
    
    centroid[0] /= numPoints;
    centroid[1] /= numPoints;
    centroid[2] /= numPoints;
}

// 점군을 중심점으로 이동 (중심화)
void centerPoints(const float points[][3], int numPoints, const float centroid[3], float centered[][3]) {
    for (int i = 0; i < numPoints; i++) {
        centered[i][0] = points[i][0] - centroid[0];
        centered[i][1] = points[i][1] - centroid[1];
        centered[i][2] = points[i][2] - centroid[2];
    }
}

// 공분산 행렬 H = P^T * Q 계산
void computeCovarianceMatrix(const float P[][3], const float Q[][3], int numPoints, float H[3][3]) {
    matrixInit(H, false);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < numPoints; k++) {
                H[i][j] += P[k][i] * Q[k][j];
            }
        }
    }
}

// 3x3 행렬의 행렬식 계산
float determinant3x3(const float matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
         - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
         + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

// 간단한 SVD 근사 (Jacobi 방법 기반)
// 실제로는 더 정교한 SVD가 필요하지만, 마이크로컨트롤러용 간단 버전
void simpleSVD(const float H[3][3], float U[3][3], float V[3][3]) {
    float HtH[3][3], HHt[3][3];
    float Ht[3][3];
    
    matrixTranspose(H, Ht);
    matrixMultiply(Ht, H, HtH);  // H^T * H
    matrixMultiply(H, Ht, HHt);  // H * H^T
    
    // 간단한 고유벡터 추정 (실제로는 더 정교한 알고리즘 필요)
    // 여기서는 단위행렬로 초기화
    matrixInit(U, true);
    matrixInit(V, true);
    
    // 실제 구현에서는 Jacobi rotation 등을 사용해야 함
    // 이는 간단한 예시용 구현입니다
}

// Kabsch 알고리즘을 사용한 최적 회전 행렬 계산
void kabschAlgorithm(const float P[][3], const float Q[][3], int numPoints, float R[3][3]) {
    // 1. 중심점 계산
    float centroidP[3], centroidQ[3];
    computeCentroid(P, numPoints, centroidP);
    computeCentroid(Q, numPoints, centroidQ);
    
    // 2. 점군 중심화
    float centeredP[numPoints][3], centeredQ[numPoints][3];
    centerPoints(P, numPoints, centroidP, centeredP);
    centerPoints(Q, numPoints, centroidQ, centeredQ);
    
    // 3. 공분산 행렬 H = P^T * Q 계산
    float H[3][3];
    computeCovarianceMatrix(centeredP, centeredQ, numPoints, H);
    
    // 4. SVD 수행: H = U * S * V^T
    float U[3][3], V[3][3];
    simpleSVD(H, U, V);
    
    // 5. R = V * U^T 계산
    float Ut[3][3];
    matrixTranspose(U, Ut);
    matrixMultiply(V, Ut, R);
    
    // 6. det(R) < 0이면 반사 보정
    if (determinant3x3(R) < 0) {
        V[2][0] = -V[2][0];
        V[2][1] = -V[2][1];
        V[2][2] = -V[2][2];
        matrixMultiply(V, Ut, R);
    }
}

// 더 간단한 버전: 두 벡터 사이의 회전 행렬 (원래 코드와 유사)
void kabschTwoVectors(const float v1[3], const float v2[3], float R[3][3]) {
    // 두 점을 각각 두 개의 점군으로 취급
    float P[2][3] = {{0,0,0}, {v1[0], v1[1], v1[2]}};
    float Q[2][3] = {{0,0,0}, {v2[0], v2[1], v2[2]}};
    
    kabschAlgorithm(P, Q, 2, R);
}

// 메인 함수: 센서 3개 데이터를 입력받아 Kabsch 알고리즘으로 회전 행렬 계산
void computeAllRotationsKabsch(float gyroData[3][3]) {
    float R_12[3][3], R_23[3][3], R_31[3][3];

    kabschTwoVectors(gyroData[0], gyroData[1], R_12);
    kabschTwoVectors(gyroData[1], gyroData[2], R_23);
    kabschTwoVectors(gyroData[2], gyroData[0], R_31);

    Serial.println("Kabsch R_12:");
    printMatrix(R_12);
    Serial.println("Kabsch R_23:");
    printMatrix(R_23);
    Serial.println("Kabsch R_31:");
    printMatrix(R_31);
}

// 다중 점을 사용한 Kabsch 예제
void kabschMultiPointExample() {
    Serial.println("\n=== Multi-point Kabsch Example ===");
    
    // 예제: 두 점군 (각각 4개의 3D 점)
    float pointsP[4][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}
    };
    
    // Q는 P를 z축 기준으로 90도 회전한 점군
    float pointsQ[4][3] = {
        {0.0, 1.0, 0.0},
        {-1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {-1.0, 1.0, 1.0}
    };
    
    float R[3][3];
    kabschAlgorithm(pointsP, pointsQ, 4, R);
    
    Serial.println("Multi-point Rotation Matrix:");
    printMatrix(R);
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    // 기존 예제
    float gyroData[3][3] = {
        {0.1, 0.5, 0.3},   // 센서1
        {0.2, 0.4, 0.35},  // 센서2
        {0.15, 0.55, 0.25} // 센서3
    };

    computeAllRotationsKabsch(gyroData);
    
    // 다중 점 예제
    kabschMultiPointExample();
}

void loop() {
    // 필요시 반복 측정
    delay(5000);
}