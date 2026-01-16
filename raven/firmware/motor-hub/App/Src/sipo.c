#include "sipo.h"
#include "main.h"

void Sipo_Init(struct Sipo_Handle* handle) {
	handle->cycle = 0;
	handle->data = 0;
}

void Sipo_Update(struct Sipo_Handle* handle) {
	const uint8_t parity = handle->cycle & 1;
	HAL_GPIO_WritePin(handle->clkPort, handle->clkPin, parity);
	if (!handle->cycle) {
		HAL_GPIO_WritePin(handle->loadPort, handle->loadPin, 1);
	}
	if (!parity) {
		HAL_GPIO_WritePin(handle->dataPort, handle->dataPin, (handle->data >> (7 - (handle->cycle>>1))) & 1);
	}
	if (handle->cycle == 15) {
		HAL_GPIO_WritePin(handle->loadPort, handle->loadPin, 0);
	}
	handle->cycle++;
	handle->cycle &= 0x0F;
}
