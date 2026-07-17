import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Marker, Callout } from 'react-native-maps';

export function LandmarkPin({ landmark, onPress }) {
  return (
    <Marker
      coordinate={{ latitude: landmark.latitude, longitude: landmark.longitude }}
      onPress={() => onPress(landmark)}
    >
      <View style={styles.pin}>
        <Text style={styles.icon}>🏛</Text>
      </View>
      <Callout tooltip>
        <View style={styles.callout}>
          <Text style={styles.calloutTitle}>{landmark.name}</Text>
          {landmark.description ? (
            <Text style={styles.calloutDesc} numberOfLines={2}>
              {landmark.description}
            </Text>
          ) : null}
          {landmark.content_count > 0 ? (
            <Text style={styles.audioTag}>🎧 Audio available</Text>
          ) : null}
        </View>
      </Callout>
    </Marker>
  );
}

const styles = StyleSheet.create({
  pin: {
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 4,
    borderWidth: 1.5,
    borderColor: '#2563EB',
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  icon: { fontSize: 18 },
  callout: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 10,
    maxWidth: 200,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 5,
  },
  calloutTitle: { fontWeight: '700', fontSize: 14, color: '#111827', marginBottom: 2 },
  calloutDesc:  { fontSize: 12, color: '#6B7280' },
  audioTag:     { fontSize: 11, color: '#2563EB', marginTop: 4, fontWeight: '600' },
});
