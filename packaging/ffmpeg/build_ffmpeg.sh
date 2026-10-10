#!/usr/bin/env bash
# Compila un ffmpeg mínimo para musicviz: sólo los códecs, formatos, filtros y protocolos que usa la app.
#
#   build_ffmpeg.sh <carpeta con el código fuente de ffmpeg> <carpeta de instalación> [opciones extra de configure]
#
# Lo que necesita musicviz:
#   - exportar: h264 (NVENC o libx264), hevc NVENC, aac, mov/mp4, y los formatos con alfa (prores_ks, qtrle,
#     vp9 en webm con libopus, secuencia PNG)
#   - leer el audio del proyecto (mp3, aac/m4a, flac, ogg/opus, wav, aiff, wma) y volcarlo como f32le por tubería
#   - decodificar el video de fondo a RGB por tubería: h264, hevc, vp8/vp9, mpeg4, prores, mjpeg, gif...
#   - fuentes lavfi (color, testsrc2, sine, concat): la app comprueba los encoders con `-f lavfi -i color=...`
#     y las pruebas generan clips sintéticos con ellas
# En Windows se enlaza estático (un único ffmpeg.exe de ~20 MB en vez de los ~80 MB del paquete "essentials").
set -euo pipefail
SRC=${1:?ruta al código fuente de ffmpeg}
PREFIX=${2:?carpeta de instalación}
shift 2

DECODERS=h264,hevc,vp8,vp9,mpeg4,mpeg2video,mpeg1video,msmpeg4v3,wmv2,theora,prores,mjpeg,png,gif,rawvideo,wrapped_avframe
DECODERS+=,mp3,mp3float,aac,aac_latm,flac,vorbis,opus,alac,wmav2,ac3,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_u8,pcm_s16be,pcm_s24be
ENCODERS=libx264,aac,libopus,prores_ks,libvpx_vp9,png,qtrle,rawvideo,wrapped_avframe,pcm_f32le,pcm_s16le
DEMUXERS=mov,matroska,avi,mpegts,mpegps,mpegvideo,image2,mp3,aac,flac,ogg,wav,aiff,asf,rawvideo,gif
MUXERS=mp4,mov,ipod,matroska,webm,image2,null,rawvideo,pcm_f32le,pcm_s16le,wav
PARSERS=h264,hevc,vp8,vp9,mpeg4video,mpegvideo,mpegaudio,aac,aac_latm,flac,vorbis,opus,png,mjpeg,ac3
BSFS=aac_adtstoasc,h264_mp4toannexb,hevc_mp4toannexb,extract_extradata,vp9_superframe,vp9_superframe_split,null
FILTERS=scale,format,fps,null,anull,aformat,aresample,volume,trim,atrim,setpts,asetpts,copy,acopy,apad
FILTERS+=,color,testsrc,testsrc2,sine,anullsrc,aevalsrc,concat,amix

cd "$SRC"
./configure --prefix="$PREFIX" \
  --disable-everything --disable-autodetect --disable-doc --disable-debug --disable-ffplay --disable-ffprobe \
  --disable-network --disable-shared --enable-static --enable-gpl --pkg-config-flags=--static \
  --enable-zlib --enable-libx264 --enable-libvpx --enable-libopus \
  --enable-swscale --enable-swresample --enable-avfilter \
  --enable-avdevice --enable-indev=lavfi \
  --enable-protocol=file,pipe,fd \
  --enable-decoder="$DECODERS" --enable-encoder="$ENCODERS" \
  --enable-demuxer="$DEMUXERS" --enable-muxer="$MUXERS" \
  --enable-parser="$PARSERS" --enable-bsf="$BSFS" --enable-filter="$FILTERS" \
  "$@"
make -j"$(nproc)"
make install
