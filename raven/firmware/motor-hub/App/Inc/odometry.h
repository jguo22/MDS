#ifndef INC_ODOMETRY_H_
#define INC_ODOMETRY_H_

#include "encoder.h"

struct Odometry_Handle{
	// configuration
	float wheelDiameter;
	float baseDiameter;
	float ticksPerRevolution;

	// internal
	float x;
	float y;
	float angle;

	// calculated internal variables
	float dTheta;
	float xf;
	float yf;
};

void Odometry_Init(struct Odometry_Handle* handle);
void Odometry_CalculateConstants(struct Odometry_Handle* handle);

// recommended 2 * (state changes / sec)
void Odometry_UpdateLeft(struct Odometry_Handle* handle, enum Encoder_Event event);
void Odometry_UpdateRight(struct Odometry_Handle* handle, enum Encoder_Event event);

#endif