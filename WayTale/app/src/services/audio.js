/**
 * Audio playback service using expo-av.
 * Plays pre-cached audio files by URL.
 * Handles play/pause/resume for the live tour.
 */

import { Audio } from 'expo-av';

let sound = null;

export async function configureAudio() {
  await Audio.setAudioModeAsync({
    playsInSilentModeIOS: true,
    staysActiveInBackground: true,
    shouldDuckAndroid: true,
  });
}

export async function playAudio(uri) {
  await stopAudio();

  const { sound: newSound } = await Audio.Sound.createAsync(
    { uri },
    { shouldPlay: true }
  );
  sound = newSound;
  return sound;
}

export async function pauseAudio() {
  if (sound) {
    const status = await sound.getStatusAsync();
    if (status.isPlaying) await sound.pauseAsync();
  }
}

export async function resumeAudio() {
  if (sound) {
    const status = await sound.getStatusAsync();
    if (!status.isPlaying) await sound.playAsync();
  }
}

export async function stopAudio() {
  if (sound) {
    await sound.stopAsync();
    await sound.unloadAsync();
    sound = null;
  }
}

export async function getPlaybackStatus() {
  if (!sound) return null;
  return sound.getStatusAsync();
}
