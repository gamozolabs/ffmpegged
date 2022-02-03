use std::fs::File;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::path::Path;
use ac_ffmpeg::codec::{AudioCodecParameters};
use ac_ffmpeg::codec::{Encoder, Decoder, VideoCodecParameters};
use ac_ffmpeg::codec::video::{VideoDecoder, VideoEncoder, PixelFormat};
use ac_ffmpeg::packet::Packet;
use ac_ffmpeg::format::io::{MemWriter, IO};
use ac_ffmpeg::format::muxer::{Muxer, OutputFormat};
use ac_ffmpeg::format::stream::Stream;
use ac_ffmpeg::format::demuxer::Demuxer;
use indicatif::ProgressBar;

/// Wrapper around [`Error`]
type Result<T> = std::result::Result<T, Error>;

/// Errors for this module
#[derive(Debug)]
pub enum Error {
    /// Failed to open input file
    Open(std::io::Error),

    /// Failed to join worker thread
    JoinThread,

    /// Failed to build demuxer
    BuildDemuxer(ac_ffmpeg::Error),

    /// Failed to find stream info for a given input
    FindDemuxerInfo(ac_ffmpeg::Error),

    /// Failed to create decoder for video content
    CreateVideoDecoder(ac_ffmpeg::Error),
    
    /// Failed to create encoder for video content
    CreateVideoEncoder(ac_ffmpeg::Error),
    
    /// Could not find the requested codec for encoding
    InvalidEncoderCodec(ac_ffmpeg::Error),
    
    /// Unknown pixel format selected for encoding
    UnknownPixelFormat(ac_ffmpeg::codec::video::frame::UnknownPixelFormat),
    
    /// Failed to push data to muxer
    MuxerPush(ac_ffmpeg::Error),
    
    /// Failed to push data to encoder
    EncodePush(ac_ffmpeg::Error),

    /// Failed to get a frame from the encoder
    EncodeTake(ac_ffmpeg::Error),

    /// Failed to push data to decoder
    DecodePush(ac_ffmpeg::Error),

    /// Failed to get a frame from the decoder
    DecodeTake(ac_ffmpeg::Error),
    
    /// Failed to get a packet from the demuxer
    DemuxTake(ac_ffmpeg::Error),

    /// Input file had multiple video streams (not supported)
    MultipleVideoStreams,

    /// Input file had multiple audio streams (not supported)
    MultipleAudioStreams,

    /// Input file had no video stream
    NoVideoStream,

    /// Input file had no audio stream
    NoAudioStream,
}

pub struct Ffmpegged {
    keyframes: Vec<usize>,
    packets: Vec<Packet>,
    vparam: VideoCodecParameters,
    aparam: AudioCodecParameters,
    decode: AtomicUsize,
    decode_progress: ProgressBar,
}

impl Ffmpegged {
    /// Demux an input file into RAM
    pub fn load(filename: impl AsRef<Path>) -> Result<Self> {
        // Open the input file
        let input = File::open(filename).map_err(Error::Open)?;

        // Create stream to the input
        let io = IO::from_seekable_read_stream(input);

        // Get the appropriate demuxer for the input
        let mut demuxer = Demuxer::builder().build(io)
            .map_err(Error::BuildDemuxer)?
            .find_stream_info(None)
            .map_err(|(_, e)| Error::FindDemuxerInfo(e))?;

        // Get the video and audio streams, we're strict for now and only
        // allow one video, one audio, and no other streams
        let mut video = None;
        let mut audio = None;
        for (stream_id, stream) in demuxer.streams().iter().enumerate() {
            // Get the codec parameters for the stream
            let params = stream.codec_parameters();

            // Determine the codec type
            if params.is_video_codec() {
                // Fail on multiple video streams
                if video.is_some() {
                    return Err(Error::MultipleVideoStreams);
                }

                // Set the video stream
                video = Some((stream_id, stream,
                    params.into_video_codec_parameters().unwrap()));
            } else if params.is_audio_codec() {
                // Fail on multiple audio streams
                if audio.is_some() {
                    return Err(Error::MultipleAudioStreams);
                }

                // Set the audio stream
                audio = Some((stream_id, stream,
                    params.into_audio_codec_parameters().unwrap()));
            } else {
                // We don't handle files with anything other than 1 audio
                // and 1 video stream
                unimplemented!("Got unexpected stream type");
            }
        }

        // Make sure we have both an audio and video stream
        let (video_idx, video, vparam) = video.ok_or(Error::NoVideoStream)?;
        let (audio_idx, audio, aparam) = audio.ok_or(Error::NoAudioStream)?;

        // Create progress bar for initial file loading
        let pb = if let Some(frames) = video.frames() {
            // We know the number of frames so make a normal progress bar
            let pb = ProgressBar::new(frames);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} {msg:20} [{wide_bar:.cyan/blue}] frame {pos} of {len} @ {per_sec} ({eta})")
                .progress_chars("#>-"),
            );
            pb
        } else {
            // Unknown number of frames, just make a spinner
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} {msg:20}")
            );
            pb
        };

        pb.set_message("Loading input");

        // Limit progress bar updates to 10 per second, this dramatically
        // reduces progress bar updating overhead
        pb.set_draw_rate(10);

        // Load the video stream from the file
        let mut packets = Vec::new();
        let mut keyframes = Vec::new();
        while let Some(packet) = demuxer.take().map_err(Error::DemuxTake)? {
            // Skip packets not related to the video steam
            if packet.stream_index() != video_idx {
                continue;
            }

            // Save keyframe metadata
            if packet.is_key() {
                keyframes.push(packets.len());
            }

            // Save the packet
            packets.push(packet);

            // Update progress bar
            pb.inc(1);
        }

        // Done with the progress bar
        pb.finish_at_current_pos();
        println!();

        // Reset and repurpose the progress bar for decoding
        pb.reset();
        pb.reset_eta();
        pb.reset_elapsed();
        pb.set_message("Decoding input");

        Ok(Self {
            keyframes, packets, vparam, aparam,
            decode: AtomicUsize::new(0),
            decode_progress: pb,
        })
    }

    pub fn decode_worker(self: Arc<Self>) -> Result<()> {
        // Create a decoder for this thread
        let mut decoder = VideoDecoder::from_codec_parameters(&self.vparam)
            .map_err(Error::CreateVideoDecoder)?
            .build()
            .map_err(Error::CreateVideoDecoder)?;

        // Create codec parameters for our encoder
        let codec_params = VideoCodecParameters::builder("libx264")
            .map_err(Error::InvalidEncoderCodec)?
            .pixel_format(
                PixelFormat::from_str("yuv420p")
                .map_err(Error::UnknownPixelFormat)?)
            .width(self.vparam.width())
            .height(self.vparam.height())
            .bit_rate(10000000)
            .build();

        // Create encoder
        let mut encoder = VideoEncoder::from_codec_parameters(&codec_params)
            .map_err(Error::CreateVideoEncoder)?
            .build()
            .map_err(Error::CreateVideoEncoder)?;

        let mut tmp = File::create("output.mp4").unwrap();
        let iow = IO::from_seekable_write_stream(tmp);

        let mut muxer = Muxer::builder();
        muxer.add_stream(&codec_params.clone().into()).unwrap();
        let mut muxer = muxer.build(iow, OutputFormat::find_by_name("mp4").unwrap()).unwrap();

        loop {
            // Get a unique ID for the keyframe chunk to process
            let chunk = self.decode.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.keyframes.len() {
                // All done!
                break;
            }

            // Get the keyframe we are to process
            let keyframe = self.keyframes[chunk];

            // Get the packets we should process
            let packets = if let Some(&end) = self.keyframes.get(chunk + 1) {
                // Process a single keyframe and incremental frames until the
                // next keyframe
                &self.packets[keyframe..end]
            } else {
                // End of stream, consume from keyframe to end
                &self.packets[keyframe..]
            };

            // Process packets
            for packet in packets {
                // Queue up this packet for decoding
                decoder.push(packet.clone()).map_err(Error::DecodePush)?;

                // Consume frames
                while let Some(frame) =
                        decoder.take().map_err(Error::DecodeTake)? {
                    // Encode the frame
                    encoder.push(frame).map_err(Error::EncodePush)?;

                    // Get all packets from the encoder
                    while let Some(enc_packet) = encoder.take()
                            .map_err(Error::EncodeTake)? {
                        // Save the encoded packet, we'll reorder these later
                        // for writing to the file
                        muxer.push(enc_packet).map_err(Error::MuxerPush)?;
                    }

                    self.decode_progress.inc(1);
                }
            }
        }

        Ok(())
    }

    /// Perform parallel decoding of the input
    pub fn decode(self: Arc<Self>, num_threads: usize) -> Result<()> {
        // Spawn worker threads
        let mut workers = Vec::new();
        for _ in 0..num_threads {
            let sc = self.clone();
            workers.push(std::thread::spawn(move || sc.decode_worker()));
        }

        // Wait for all threads to complete
        for worker in workers {
            worker.join().map_err(|_| Error::JoinThread)??;
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    let ff = Arc::new(Ffmpegged::load("../streamvods/out.mp4")?);
    //let ff = Arc::new(Ffmpegged::load("../streamvods/2022-02-02_09-57-09.mp4")?);
    //let ff = Arc::new(Ffmpegged::load("../streamvods/2022-01-30_21-26-45.mp4")?);
    ff.decode(1)?;
    Ok(())
}

