# Hackathon 2023-03-31
## Aufgabe
Es gibt 4 Stationen. Sobi soll zu erst die Farbe jeweilige Station erkennen und speichen, dannach lagert Sobi die zupassende Guete nach Farbe in Lagertische. Yu transportiert Guete zu Sobi. Ausschliesslich Sobi transportiert die Guete zurueck zu Stationen. 
## Objekterkennung
Meine Aufgabe ist Erkennung der Farbe und Formung(Dreieck, Kreis, Stern) durch klassischen Bildverarbeitung. Die Code wird auf Python geschireben. Das Ergebnis wird durch Topic von ROS veroeffentlicht, um StateMaschine zu zugreiffen.  objErkennung.py ist erste Version, die filters der Kontours seperat sind, ist es besser fuer Debug und die Kpntours werden nur durch grau Bild extraiert. objErkennungGray.py ist meist mit die erste Version identisch aber die filters werden zusammen gesetzt wegen der Verbesserung der Leistungsanforderung. objErkennunglabMitGary.py hat die a und b von lab Farbraum genutzt, um Kontours zu finden, weil die Formungen, die durch gelb gefuellt wird, nicht unter garu Bildgefunden werden. 
## DatenStruktur
'''MultiArrayUnit32''' als grunde Datentype. die Lanege davon ist 10. Die erste fuenf Werten sind die Formung des Objekt, und die andere 5 Werten bezeichnen sich die Farben. 
## Video
[Youtube](https://youtu.be/eZwafD0QqKs)
