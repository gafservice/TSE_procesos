SUMMARY = "bitbake-layers recipe"
DESCRIPTION = "Recipe created by bitbake-layers"
LICENSE = "CLOSED"
LIC_FILES_CHKSUM = ""


SRC_URI = "file://iniciador.py \
	   file://model.tflite "
	   
	
S = "${WORKDIR}"
TARGER_CC_ARCH += "${LDFLAGS}"
MY_DESTINATION = "/home/root/tsep02"
do_install(){
    install -d ${D}${MY_DESTINATION}
    install -m 0755 iniciador.py ${D}${MY_DESTINATION}
}  
do_install_append() { 
    install -d ${D}${MYDESTINATION}
    install -m 0755 model.tflite ${D}${MY_DESTINATION}  
}

FILES_${PN} += "${MY_DESTINATION}/*"

#pkg_postinst_ontarget_iniciador(){
	#!/bin/sh
#	python3 /usr/bin/iniciador.py
#	sleep 5
#	poweroff
#}


