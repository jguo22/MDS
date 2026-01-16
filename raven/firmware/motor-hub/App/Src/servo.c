#include "servo.h"

void Servo_Init(struct Servo_Handle* handle) {
    HAL_TIM_PWM_Start(handle->tim, handle->chan);
    __HAL_TIM_SET_COMPARE(handle->tim, handle->chan, 0);
}

void Servo_Write(struct Servo_Handle* handle, uint16_t pos) {
    __HAL_TIM_SET_COMPARE(handle->tim, handle->chan, pos);
}

uint16_t Servo_Read(struct Servo_Handle* handle) {
    return __HAL_TIM_GET_COMPARE(handle->tim, handle->chan);
}