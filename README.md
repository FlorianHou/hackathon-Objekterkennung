# Hackathon 2023-03-31
## Aufgabe
Es gibt 4 Stationen. Sobi soll zu erst die Farbe jeweilige Station erkennen und speichen, dannach lagert Sobi die zupassende Guete nach Farbe in Lagertische. Yu transportiert Guete zu Sobi. Ausschliesslich Sobi transportiert die Guete zurueck zu Stationen. 
## Objekterkennung
Meine Aufgabe ist Erkennung der Farbe und Formung(Dreieck, Kreis, Stern) durch klassischen Bildverarbeitung. Die Code wird auf Python geschireben. Das Ergebnis wird durch Topic von ROS veroeffentlicht, um StateMaschine zu zugreiffen. 
## DatenStruktur
'''MultiArrayUnit32''' als grunde Datentype. die Lanege davon ist 10. Die erste fuenf Werten sind die Formung des Objekt, und die andere 5 Werten bezeichnen sich die Farben. 
## Video
[Youtube](https://youtu.be/eZwafD0QqKs)
