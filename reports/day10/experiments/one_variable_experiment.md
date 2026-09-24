# Day 10 One Variable Experiment

## Window A/B 결과

- Window 5초: Prediction Updates = 0
- Window 3초: Prediction Updates = 5
- Window 3초 Mean Confidence = 0.2487
- Window 3초 UNKNOWN Ratio = 1.00

## Confidence A/B 결과

고정 조건:

- Window Seconds: 3
- Stabilization Buffer: 5

비교:

- Before Confidence Threshold: 0.55
- After Confidence Threshold: 0.20

결과:

- Prediction Updates: 5 → 5
- First Prediction: 2.44초 → 2.44초
- UNKNOWN Ratio: 1.00 → 0.00
- Stable Match Ratio: 0.00 → 0.40
- Stable Label Changes: 0 → 1
- Mean Confidence: 0.2487 → 0.2487
- Mean Processing FPS: 80.589 → 75.166

## Final Runtime Conditions

- Window Seconds: 3
- Confidence Threshold: 0.20
- Stabilization Buffer: 5

선택 근거:

- Window: 5초에서는 Prediction이 생성되지 않았고, 3초에서는 5회 생성되었다.
- Confidence: 0.55에서 UNKNOWN Ratio가 1.00이었지만 0.20에서는 0.00으로 감소했다.
- Confidence: Stable Match Ratio도 0.00에서 0.40으로 증가했다.
- Stabilization: 이번 A/B 실험에서는 5로 고정하여 비교했다.

주의:

- Confidence 0.20이 모든 상황에서 최적이라는 뜻은 아니다.
- 이번 S02 Validation A/B 결과를 기준으로 Final Challenge에서 사용할 값으로 고정한다.
- S03 Test 결과를 본 뒤 이 값을 다시 조정하지 않는다.
