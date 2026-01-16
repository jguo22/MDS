#include "encoder.h"

void Encoder_Init(struct Encoder_Handle* handle) {
	handle->aLast = 0;
	handle->bLast = 0;

	handle->pos = 0;
}

enum Encoder_Event Encoder_Update(struct Encoder_Handle* handle) {
	const bool aState = HAL_GPIO_ReadPin(handle->aPort, handle->aPin);
	const bool bState = HAL_GPIO_ReadPin(handle->bPort, handle->bPin);

	const bool aDiff = aState^handle->aLast;
	const bool bDiff = bState^handle->bLast;

	enum Encoder_Event event = EncoderNoEvent;

	if (aDiff^bDiff) {
		const uint8_t idir = aDiff ^ aState ^ bState;
		if (idir) {
			handle->pos++;
			event = EncoderIncrease;
		}else{
			handle->pos--;
			event = EncoderDecrease;
		}
	}
	handle->aLast = aState;
	handle->bLast = bState;

	return event;
}
