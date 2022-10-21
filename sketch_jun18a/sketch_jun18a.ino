#include<Servo.h>
Servo X ;
Servo Y ;

void setup() {
  
X.attach(7);
Y.attach(8);
Serial.begin(9600);
X.write(90);
Y.write(180);
}
void loop() {
if(Serial.available()){
 char x =Serial.read();
 int v =Serial.parseInt();
 
if(x =='a'){
  X.write(v);
  }
  if(x =='b'){
  Y.write(v);
  }
  }
}
