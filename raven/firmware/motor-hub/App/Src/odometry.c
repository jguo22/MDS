#include "odometry.h"
#include "math.h"

void Odometry_Init(struct Odometry_Handle* handle){
	handle->x = 0;
	handle->y = 0;
	handle->angle = 0;

	Odometry_CalculateConstants(handle);
}

void Odometry_CalculateConstants(struct Odometry_Handle* handle){
    handle->dTheta = M_PI * handle->wheelDiameter / (handle->baseDiameter * handle->ticksPerRevolution);
    handle->xf = handle->baseDiameter/2.0 * sinf(handle->dTheta);
    handle->yf = handle->baseDiameter/2.0 * (1 - cosf(handle->dTheta));
}

void Odometry_UpdateRight(struct Odometry_Handle* handle, enum Encoder_Event event){
	float cos = cosf(handle->angle);
	float sin = sinf(handle->angle);
	switch (event) {
		case EncoderIncrease:
		{
			handle->x += handle->xf * cos - handle->yf * sin;
			handle->y += handle->xf * sin + handle->yf * cos;
			handle->angle += handle->dTheta;
			break;
		}
		case EncoderDecrease:
			handle->x += -handle->xf * cos - handle->yf * sin;
			handle->y += -handle->xf * sin + handle->yf * cos;
			handle->angle -= handle->dTheta;
			break;
	}
}

void Odometry_UpdateLeft(struct Odometry_Handle* handle, enum Encoder_Event event){
	float cos = cosf(handle->angle);
	float sin = sinf(handle->angle);
	switch (event) {
		case EncoderDecrease:
		{
			handle->x += handle->xf * cos + handle->yf * sin;
			handle->y += handle->xf * sin - handle->yf * cos;
			handle->angle -= handle->dTheta;
			break;
		}
		case EncoderIncrease:
			handle->x += -handle->xf * cos + handle->yf * sin;
			handle->y += -handle->xf * sin - handle->yf * cos;
			handle->angle += handle->dTheta;
			break;
	}
}


