// Timestamped four-sensor MLX90393 readout through the PCA9548 I2C mux.
// Wiring: ESP32 GPIO21 -> SDA, GPIO22 -> SCL, both to PCA9548 (STEMMA QT).
// MLX90393 sensors are on mux channels 0, 2, 5, and 7, each at address 0x18.
// Packet:
// FRAME,sequence,device_us,s0x,s0y,s0z,...,s3x,s3y,s3z

#include <Wire.h>
#include <Adafruit_MLX90393.h>
#include <math.h>

constexpr uint8_t MUX_ADDR = 0x70;
constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t MLX90393_ADDR = 0x18;
constexpr uint8_t SENSOR_COUNT = 4;
constexpr uint8_t MUX_CHANNELS[SENSOR_COUNT] = {0, 2, 5, 7};
constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t SAMPLE_RATE_HZ = 50;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
constexpr uint32_t SENSOR_RETRY_PERIOD_US = 1000000UL;

Adafruit_MLX90393 sensors[SENSOR_COUNT];
bool sensorReady[SENSOR_COUNT] = {false, false, false, false};
uint32_t nextSensorRetryUs[SENSOR_COUNT] = {0, 0, 0, 0};
uint32_t sequenceNumber = 0;
uint32_t nextSampleUs = 0;

void muxSelectChannel(uint8_t channel) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

bool devicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

bool initializeSensor(uint8_t sensorIndex) {
  const uint8_t channel = MUX_CHANNELS[sensorIndex];
  muxSelectChannel(channel);

  if (!devicePresent(MLX90393_ADDR) ||
      !sensors[sensorIndex].begin_I2C(MLX90393_ADDR, &Wire)) {
    Serial.print(F("ERROR,sensor="));
    Serial.print(sensorIndex);
    Serial.print(F(",mux_channel="));
    Serial.print(channel);
    Serial.println(F(",not_found"));
    return false;
  }

  const bool configured =
      sensors[sensorIndex].setResolution(MLX90393_X, MLX90393_RES_16) &&
      sensors[sensorIndex].setResolution(MLX90393_Y, MLX90393_RES_16) &&
      sensors[sensorIndex].setResolution(MLX90393_Z, MLX90393_RES_16) &&
      sensors[sensorIndex].setOversampling(MLX90393_OSR_1) &&
      sensors[sensorIndex].setFilter(MLX90393_FILTER_3);
  if (!configured) {
    Serial.print(F("ERROR,sensor="));
    Serial.print(sensorIndex);
    Serial.println(F(",configuration_failed"));
    return false;
  }

  Serial.print(F("SENSOR_READY,sensor="));
  Serial.print(sensorIndex);
  Serial.print(F(",mux_channel="));
  Serial.println(channel);
  return true;
}

bool readSensor(uint8_t sensorIndex, float *x, float *y, float *z) {
  if (!sensorReady[sensorIndex]) {
    const uint32_t nowUs = micros();
    if (static_cast<int32_t>(nowUs - nextSensorRetryUs[sensorIndex]) >= 0) {
      sensorReady[sensorIndex] = initializeSensor(sensorIndex);
      nextSensorRetryUs[sensorIndex] = nowUs + SENSOR_RETRY_PERIOD_US;
    }
  }

  if (!sensorReady[sensorIndex]) {
    return false;
  }

  muxSelectChannel(MUX_CHANNELS[sensorIndex]);
  if (sensors[sensorIndex].readData(x, y, z)) {
    return true;
  }

  sensorReady[sensorIndex] = false;
  nextSensorRetryUs[sensorIndex] = micros() + SENSOR_RETRY_PERIOD_US;
  Serial.print(F("ERROR,sensor="));
  Serial.print(sensorIndex);
  Serial.println(F(",measurement_failed"));
  return false;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { delay(10); }
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);

  Serial.println();
  Serial.println(F("MLX90393 Four-Sensor Timestamped Live Readout"));
  Serial.println(F("================================================"));

  for (uint8_t sensorIndex = 0; sensorIndex < SENSOR_COUNT; ++sensorIndex) {
    sensorReady[sensorIndex] = initializeSensor(sensorIndex);
    nextSensorRetryUs[sensorIndex] = micros() + SENSOR_RETRY_PERIOD_US;
  }

  Serial.print(F("READY,protocol=FRAME_v1,address=0x"));
  Serial.print(MLX90393_ADDR, HEX);
  Serial.print(F(",sensor_count="));
  Serial.print(SENSOR_COUNT);
  Serial.print(F(",mux_channels=0|2|5|7,rate_hz="));
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(F(",filter=3,oversampling=1,resolution_bits=16,unit=uT"));
  nextSampleUs = micros();
}

void loop() {
  const int32_t untilSample = static_cast<int32_t>(nextSampleUs - micros());
  if (untilSample > 0) {
    delayMicroseconds(static_cast<uint32_t>(untilSample));
  }
  const uint32_t sampleStartedUs = micros();
  const uint32_t sampleSequence = sequenceNumber++;
  nextSampleUs += SAMPLE_PERIOD_US;

  float readings[SENSOR_COUNT][3];
  for (uint8_t sensorIndex = 0; sensorIndex < SENSOR_COUNT; ++sensorIndex) {
    float x = NAN;
    float y = NAN;
    float z = NAN;
    readSensor(sensorIndex, &x, &y, &z);
    readings[sensorIndex][0] = x;
    readings[sensorIndex][1] = y;
    readings[sensorIndex][2] = z;
  }

  Serial.print(F("FRAME,"));
  Serial.print(sampleSequence);
  Serial.print(',');
  Serial.print(sampleStartedUs);
  for (uint8_t sensorIndex = 0; sensorIndex < SENSOR_COUNT; ++sensorIndex) {
    for (uint8_t axis = 0; axis < 3; ++axis) {
      Serial.print(',');
      Serial.print(readings[sensorIndex][axis], 3);
    }
  }
  Serial.println();

  // Avoid a burst of back-to-back samples after an overrun.
  if (static_cast<int32_t>(micros() - nextSampleUs) >= 0) {
    nextSampleUs = micros() + SAMPLE_PERIOD_US;
  }
}
