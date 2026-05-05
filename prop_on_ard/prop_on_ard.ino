#include <Arduino.h>

const int encArmPinA  = 20;
const int encArmPinB  = 21;
const int encPendPinA = 19;
const int encPendPinB = 18;
const int motArmPWM   = 5;
const int motArmDir   = 4;

const float PEND_SIGN  = -1.0;
const float ARM_SIGN   =  1.0;
const float MOTOR_SIGN =  1.0;

const float SUPPLY_VOLTAGE = 12.0;
const int   ENC_CPR        = 2000;
const float TICK_TO_RAD    = (2.0 * PI) / ENC_CPR;

const int   LOOP_HZ        = 100;
const float DT             = 1.0 / LOOP_HZ;
const int   SAFETY_TIMEOUT = 100; 

const float DEADZONE = 0.5;
const float THRESH   = 0.5;

// Hybrid logic boundaries
const float BLEND_INNER = 10.0 * (PI / 180.0); // 10 degrees in radians
const float BLEND_OUTER = 20.0 * (PI / 180.0); // 20 degrees in radians

// LQR Gains
float k1 = -4.4681 * 0.5;
float k2 = -1.6210 * 0.5; 
float k3 = -39.5037 * 1.0; 
float k4 = -5.0402 * 0.75; 

struct StatePacket {
  float thetaPend;
  float thetaArm;
  float omegaPend;
  float omegaArm;
};

volatile long countArm  = 0;
volatile long countPend = 0;
long  lastCountArm      = 0;
long  lastCountPend     = 0;
float omegaArmFiltered  = 0.0;
float omegaPendFiltered = 0.0;

volatile bool tickFlag  = false;
unsigned long lastCommandTime = 0;
float python_voltage = 0.0;

void isrArmA()  { (digitalRead(encArmPinA) != digitalRead(encArmPinB))  ? countArm++  : countArm--;  }
void isrArmB()  { (digitalRead(encArmPinA) == digitalRead(encArmPinB))  ? countArm++  : countArm--;  }
void isrPendA() { (digitalRead(encPendPinA) != digitalRead(encPendPinB)) ? countPend-- : countPend++; }
void isrPendB() { (digitalRead(encPendPinA) == digitalRead(encPendPinB)) ? countPend-- : countPend++; }

void applyVoltage(float volts);
void stopMotor();

void setupTimer1() {
  noInterrupts();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A  = 624;
  TCCR1B = (1 << WGM12) | (1 << CS12);
  TIMSK1 = (1 << OCIE1A);
  interrupts();
}
ISR(TIMER1_COMPA_vect) { tickFlag = true; }

void setup() {
  Serial.begin(500000);

  pinMode(encArmPinA,  INPUT_PULLUP);
  pinMode(encArmPinB,  INPUT_PULLUP);
  pinMode(encPendPinA, INPUT_PULLUP);
  pinMode(encPendPinB, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(encArmPinA),  isrArmA,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(encArmPinB),  isrArmB,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(encPendPinA), isrPendA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encPendPinB), isrPendB, CHANGE);

  pinMode(motArmPWM, OUTPUT);
  pinMode(motArmDir, OUTPUT);
  stopMotor();

  setupTimer1();
  lastCommandTime = millis();
}

void loop() {
  if (tickFlag) {
    tickFlag = false;

    noInterrupts();
    long cArm  = countArm;
    long cPend = countPend;
    interrupts();

    long dArm  = cArm  - lastCountArm;
    long dPend = cPend - lastCountPend;

    float rawVelArm  = (dArm  * TICK_TO_RAD) / DT;
    float rawVelPend = (dPend * TICK_TO_RAD) / DT;

    omegaArmFiltered  = 0.25 * rawVelArm  + 0.75 * omegaArmFiltered;
    omegaPendFiltered = 0.25 * rawVelPend + 0.75 * omegaPendFiltered;

    float pendAngle = cPend * TICK_TO_RAD * PEND_SIGN;
    float armAngle  = cArm * TICK_TO_RAD * ARM_SIGN;
    float pendVel   = omegaPendFiltered * PEND_SIGN;
    float armVel    = omegaArmFiltered  * ARM_SIGN;

    // 1. Send state packet exactly as Python expects
    StatePacket pkt;
    pkt.thetaPend = pendAngle;
    pkt.thetaArm  = armAngle;
    pkt.omegaPend = pendVel;
    pkt.omegaArm  = armVel;

    uint16_t syncWord = 0xABCD;
    Serial.write((byte*)&syncWord, sizeof(syncWord));
    Serial.write((byte*)&pkt, sizeof(pkt));

    // 2. Local LQR Math Transformation
    float pendLQR = pendAngle;
    if (pendLQR > 0) pendLQR = PI - pendLQR;
    else pendLQR = -1 * PI - pendLQR;
    
    // Wrap to [-PI, PI]
    while (pendLQR > PI) pendLQR -= 2.0 * PI;
    while (pendLQR < -PI) pendLQR += 2.0 * PI;

    pendLQR = -1 * pendLQR;

    // 3. Hardware Override & SAC Blend Logic
    float u_lqr = -(k1 * armAngle + k2 * armVel + k3 * pendLQR + k4 * pendVel);
    float u_rl  = python_voltage;
    float u     = 0.0;
    
    float abs_theta = fabs(pendLQR); // Use fabs() for floats

    if (abs_theta <= BLEND_INNER) {
      // Within 10 degrees: Pure Proportional (LQR)
      u = u_lqr;
    } 
    else if (abs_theta >= BLEND_OUTER) {
      // Beyond 20 degrees: Pure RL
      u = u_rl;
    } 
    else {
      // Between 10 and 20 degrees: Blended Controller
      // alpha = 0.0 at 10 deg, alpha = 1.0 at 20 deg
      //float alpha = (abs_theta - BLEND_INNER) / (BLEND_OUTER - BLEND_INNER);
      float alpha = 0.4;
      u = (alpha * u_rl) + ((1.0 - alpha) * u_lqr);
    }

    applyVoltage(u);

    lastCountArm  = cArm;
    lastCountPend = cPend;
  }

  // Read the incoming float from Python asynchronously
  if (Serial.available() >= 4) {
    while (Serial.available() >= 8) {
      uint8_t trash[4];
      Serial.readBytes(trash, 4);
    }
    Serial.readBytes((byte*)&python_voltage, 4);
    lastCommandTime = millis();
  }

  if (millis() - lastCommandTime > SAFETY_TIMEOUT) {
    stopMotor();
  }
}

void applyVoltage(float volts) {
  volts *= MOTOR_SIGN;
  if (volts >  THRESH)      volts += DEADZONE;
  else if (volts < -THRESH) volts -= DEADZONE;
  else                      volts  = 0.0;

  volts = constrain(volts, -SUPPLY_VOLTAGE, SUPPLY_VOLTAGE);
  float duty = constrain(volts / SUPPLY_VOLTAGE, -1.0, 1.0);
  int pwm = (int)(fabs(duty) * 255.0 + 0.5);
  digitalWrite(motArmDir, (duty >= 0.0) ? HIGH : LOW);
  analogWrite(motArmPWM, pwm);
}

void stopMotor() { analogWrite(motArmPWM, 0); }