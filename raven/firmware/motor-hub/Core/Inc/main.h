/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32g4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdint.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define A4_Pin GPIO_PIN_13
#define A4_GPIO_Port GPIOC
#define A5_Pin GPIO_PIN_14
#define A5_GPIO_Port GPIOC
#define B5_Pin GPIO_PIN_15
#define B5_GPIO_Port GPIOC
#define RSV_Pin GPIO_PIN_10
#define RSV_GPIO_Port GPIOB
#define RCLK_Pin GPIO_PIN_11
#define RCLK_GPIO_Port GPIOB
#define RDAT_Pin GPIO_PIN_12
#define RDAT_GPIO_Port GPIOB
#define PH5_Pin GPIO_PIN_13
#define PH5_GPIO_Port GPIOB
#define PWR_Pin GPIO_PIN_8
#define PWR_GPIO_Port GPIOA
#define B3_Pin GPIO_PIN_13
#define B3_GPIO_Port GPIOA
#define A3_Pin GPIO_PIN_14
#define A3_GPIO_Port GPIOA
#define SL5_Pin GPIO_PIN_15
#define SL5_GPIO_Port GPIOA
#define B2_Pin GPIO_PIN_3
#define B2_GPIO_Port GPIOB
#define A2_Pin GPIO_PIN_4
#define A2_GPIO_Port GPIOB
#define A1_Pin GPIO_PIN_5
#define A1_GPIO_Port GPIOB
#define B1_Pin GPIO_PIN_6
#define B1_GPIO_Port GPIOB
#define B4_Pin GPIO_PIN_7
#define B4_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
