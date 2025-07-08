'''
David Dalmazzo - 2025
KTH Royal Institute of Technology
AIMIR Project
This class is used to extract the form of a song using structural analysis
'''
import os
import json
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from scipy.ndimage import median_filter
from sklearn.cluster import AgglomerativeClustering

# Define the ChordChange class
class ChordChange:
    def __init__(self, chord, timestamp):
        self.chord = chord
        self.timestamp = timestamp

    def __repr__(self):
        return f"ChordChange(chord='{self.chord}', timestamp={self.timestamp})"
    
class formExtractor():
    
    def __init__(self):
        self.y = None
        self.sr = None
        self.chords = None
        self.bars = 0
             
        self.data_dict = {
            'sr': 1,
            'chords': {}, 
            'beats': {}, 
            'bound_times': np.array([0,0,0,0]), 
            'bound_frames': np.array([0,0,0,0]), 
            'bound_segments': [0,0,0,0] 
        }
    
    #--------------------------------------------------------
    #Get the data
    def getData(self, audio_path):
        #Populate the data 
        self.y, self.sr = self.loadAudio(audio_path)
        self.chords = self.getChords(audio_path)
        self.bars = self.getBars(audio_path)
        
    #--------------------------------------------------------
    #Extract the chords
    def getChords(self, audio_path):
        """
        Placeholder for chord extraction functionality.
        This method will be replaced with an alternative to the deprecated chord_extractor.
        """
        print("Note: Chord extraction method has been updated. Using empty chord dictionary for now.")
        return {}
    
    #--------------------------------------------------------
    #Load the audio file
    def loadAudio(self, audio_path):
        #Load the audio file
        y, sr = librosa.load(audio_path)
        return y, sr
    
    #--------------------------------------------------------
    #Extract the bars
    def getBars(self, audio_path):
        #Load the audio file
        y, sr = librosa.load(audio_path)
        #Extract the tempo and beat frames
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        print(f'Tempo: {tempo}')
        #Convert the beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        #Extract the bars
        # Calculate bar positions (assuming 4 beats per bar)
        bars = []
        bar_duration = 4 * 60 / tempo  # duration of one bar in seconds

        # Iterate over beat_times and group into bars
        current_bar = []
        for beat_time in beat_times:
            current_bar.append(beat_time)
            if len(current_bar) == 4:
                bars.append(current_bar)
                current_bar = []
        
        return bars
    
    #--------------------------------------------------------
    #Get the y
    def get_y(self):
        return self.y
    
    #--------------------------------------------------------
    #get the sr
    def get_sr(self):
        return self.sr
    
    #--------------------------------------------------------
    def amplitud_to_db(self, y, sr, plotIt=False):
        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)), ref=np.max)
        if plotIt:
            #define the size
            plt.figure(figsize=(20, 8))
            fig, ax = plt.subplots()
            librosa.display.specshow(C, y_axis='cqt_hz', sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', ax=ax)
        return C
    
    #-------------------------------------------------------
    #To reduce dimensionality, we'll beat-synchronous the CQT
    def sync(self, y, sr, C, plotIt=False):
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0), sr=sr)
        
        if plotIt:
            # For plotting purposes, we'll need the timing of the beats
            # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
            #define the size
            plt.figure(figsize=(20, 8))
            fig, ax = plt.subplots()
            librosa.display.specshow(Csync, bins_per_octave=12*3, y_axis='cqt_hz', x_axis='time', x_coords=beat_times, ax=ax)
        return Csync, beats, beat_times
    
    #-------------------------------------------------------
    def laplacian_2(self, y, sr, C, Csync, beats, beat_times, K, plot_it=False, threshold=0.0, min_duration=5.0):
        # Let's build a weighted recurrence matrix using beat-synchronous CQT (Equation 1)
        # width=3 prevents links within the same bar; mode='affinity' implements S_rep (after Eq. 8)
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)

        # Apply a threshold to eliminate weak similarities in R
        print(f'Threshold: {threshold}')
        R[R < threshold] = 0

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        # Build the sequence matrix using MFCC similarity
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        # Compute the balanced combination
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path

        # Apply a threshold to eliminate weak similarities in A
        A[A < threshold] = 0

        # Plot the resulting graphs (Figure 1, left and center)
        if plot_it:
            fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
            librosa.display.specshow(Rf, cmap='coolwarm', y_axis='time', x_axis='s',
                                    y_coords=beat_times, x_coords=beat_times, ax=ax[0])
            ax[0].set(title='Recurrence similarity')
            ax[0].label_outer()

            librosa.display.specshow(R_path, cmap='coolwarm', y_axis='time', x_axis='s',
                                    y_coords=beat_times, x_coords=beat_times, ax=ax[1])
            ax[1].set(title='Path similarity')
            ax[1].label_outer()

            librosa.display.specshow(A, cmap='coolwarm', y_axis='time', x_axis='s',
                                    y_coords=beat_times, x_coords=beat_times, ax=ax[2])
            ax[2].set(title='Combined graph')
            ax[2].label_outer()
            plt.show()

        # Now let's compute the normalized Laplacian (Eq. 10)
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # Cumulative normalization is needed for symmetric normalized Laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        k = K
        X = evecs[:, :k] / Cnorm[:, k-1:k]

        # Check for NaN or infinite values in X and clean them
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X)

        # Let's use these k components to cluster beats into segments (Algorithm 1)
        KM = KMeans(n_clusters=k, n_init=10)
        seg_ids = KM.fit_predict(X)

        if plot_it:
            # Plot the structure components and estimated labels
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
            colors = plt.get_cmap('coolwarm', k)

            librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
            ax[0].set(title='Structure components')

            # Prepare the segmentation labels for plotting
            img = ax[1].imshow(seg_ids[np.newaxis, :], aspect='auto', cmap=colors,
                            extent=[beat_times[0], beat_times[-1], 0, 1])
            ax[1].set(title='Estimated labels before merging')
            ax[1].set_yticks([])
            ax[1].set_xlabel('Time (s)')
            fig.colorbar(img, ax=[ax[1]], ticks=range(k))
            plt.show()

        # Define the sections
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Include beat 0 as a boundary
        bound_beats = np.concatenate(([0], bound_beats))

        # Ensure bound_beats are within valid range
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0, x_max=len(beats)-1, pad=False)

        # Compute the segment label for each boundary
        bound_segments = seg_ids[bound_beats]

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Get the total duration of the song
        total_duration = librosa.get_duration(y=y, sr=sr)

        # Ensure that bound_times includes the end of the song
        bound_times = librosa.frames_to_time(bound_frames, sr=sr)
        if bound_times[-1] < total_duration:
            bound_times = np.append(bound_times, total_duration)
            bound_frames = np.append(bound_frames, C.shape[1] - 1)
            bound_segments = np.append(bound_segments, bound_segments[-1])

        # Implement Minimum Duration Threshold
        new_bound_frames = [bound_frames[0]]
        new_bound_segments = [bound_segments[0]]

        for i in range(1, len(bound_frames)):
            # Calculate the duration of the current segment
            duration = bound_times[i] - bound_times[i - 1]
            if duration < min_duration:
                # Merge with the previous segment by not adding a new boundary
                #print(f"Merging segment {i} (duration {duration:.2f}s) with previous segment.")
                continue
            else:
                # Keep the boundary
                new_bound_frames.append(bound_frames[i])
                new_bound_segments.append(bound_segments[i])

        # Identify unique labels in the order they first appear
        unique_labels = []
        for label in new_bound_segments:
            if label not in unique_labels:
                unique_labels.append(label)

        # Create a mapping from old labels to new labels starting from 1
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        # Apply the mapping to new_bound_segments to generate final_bound_segments
        final_bound_segments = [label_mapping[label] for label in new_bound_segments]

        # Plot the final segmentation result
        if plot_it:
            fig, ax = plt.subplots(figsize=(12, 2))
            colors = plt.get_cmap('coolwarm', len(unique_labels) + 1)

            # Create arrays of segment starts and ends
            segment_starts = librosa.frames_to_time(new_bound_frames, sr=sr)
            segment_ends = np.append(segment_starts[1:], total_duration)

            # Plot each segment as a horizontal bar
            for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
                ax.broken_barh([(start, end - start)], (0, 1),
                            facecolors=colors(final_bound_segments[i]), edgecolors='black')

            ax.set_xlim([0, total_duration])
            ax.set_yticks([])
            ax.set_xlabel('Time (s)')
            ax.set_title('Final Segmentation after Merging')
            plt.show()

        return np.array(new_bound_frames), final_bound_segments
    
    #-------------------------------------------------------
    #Calculate the recurrence matrix
    def laplacian(self, y, sr, C, Csync, beats, beat_times, K, plotIt=False, threshold=0.0):
        
        # Let's build a weighted recurrence matrix using beat-synchronous CQT (Equation 1) 
        # width=3 prevents links within the same bar mode='affinity' here implements S_rep (after Eq. 8)
        
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',sym=True)

        # Apply a threshold to eliminate weak similarities in R
        
        print(f'Threshold: {threshold}')
        R[R < threshold] = 0

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        # Build the sequence matrix using MFCC similarity
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        # Compute the balanced combination
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path
        
        # Apply a threshold to eliminate weak similarities in A
        A[A < threshold] = 0
        
        #Plot the resulting graphs (Figure 1, left and center)
        if plotIt:
            fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
            librosa.display.specshow(Rf, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[0])
            ax[0].set(title='Recurrence similarity')
            ax[0].label_outer()

            librosa.display.specshow(R_path, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[1])
            ax[1].set(title='Path similarity')
            ax[1].label_outer()

            librosa.display.specshow(A, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[2])
            ax[2].set(title='Combined graph')
            ax[2].label_outer()
            
        #Now let's compute the normalized Laplacian (Eq. 10)
        
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k

        k = K
        X = evecs[:, :k] / Cnorm[:, k-1:k]
        
        # Check for NaN or infinite values in X and clean them
        if np.isnan(X).any() or np.isinf(X).any():
            X[np.isnan(X)] = 0.0
            X[np.isinf(X)] = 0.0
        
        #Let's use these k components to cluster beats into segments (Algorithm 1)
        KM = KMeans(n_clusters=k, n_init=10)

        seg_ids = KM.fit_predict(X)

        if plotIt:
            # and plot the results
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))
            colors = plt.get_cmap('coolwarm', k)

            librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
            ax[0].set(title='Structure components')

            # Convert lists to numpy arrays
            x_coords = np.array([0, 1])
            y_coords = np.array(list(beat_times) + [beat_times[-1]])

            img = librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors, y_axis='time',x_coords=x_coords, y_coords=y_coords, ax=ax[1])
            ax[1].set(title='Estimated labels')

            ax[1].label_outer()
            fig.colorbar(img, ax=[ax[1]], ticks=range(k))
            
        #define the sections
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segments = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)

        #Identify unique labels in the order they first appear
        unique_labels = []
        for label in bound_segments:
            if label not in unique_labels:
                unique_labels.append(label)

        #Create a mapping from old labels to new labels starting from 1
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        #Apply the mapping to bound_segments to generate new_bound_segments
        new_bound_segments = [label_mapping[label] for label in bound_segments]
               
        return bound_frames, new_bound_segments
            
    #-------------------------------------------------------
    #Populate the dictionary
    def populateDict(self, sr, chords, beats, bound_times, bound_frames, bound_segments):    
        #Populate the data dictionary    
        self.data_dict['sr'] = sr
        self.data_dict['chords'] = chords
        self.data_dict['beats'] = beats
        self.data_dict['bound_times'] = bound_times
        self.data_dict['bound_frames'] = bound_frames
        self.data_dict['bound_segments'] = bound_segments
        
        return self.data_dict
    
    #-------------------------------------------------------
    # Conversion function for JSON serialization
    def convert_to_serializable(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, ChordChange):
            return {'chord': data.chord, 'timestamp': data.timestamp}
        elif isinstance(data, list):
            return [self.convert_to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.convert_to_serializable(value) for key, value in data.items()}
        return data

    #-------------------------------------------------------
    # Save the data into a json file
    def saveData(self, data_dict, id_file, path, tonality=None, functional_harmony=None):
        """
        Save the data into a JSON file.
        - data_dict: The data to be saved.
        - id_file: The ID of the file.
        - path: The path to the directory where the file will be saved.
        - tonality: Optional tonality information to be included in the file.
        - functional_harmony: Optional functional harmony information to be included in the file.
        """

        # Save the data into a json file
        # Define the path to the JSON file
        name = id_file + '.json'
        myPathName = os.path.join(path, name)

        # Check if the directory exists
        if not os.path.isdir(path):
            print(f"Error: The directory '{path}' does not exist.")
            return

        # Add optional tonality and functional_harmony to the data dictionary if provided
        if tonality:
            data_dict['tonality'] = tonality
        if functional_harmony:
            data_dict['functional_harmony'] = functional_harmony

        # Convert data_dict to a JSON-serializable format
        serializable_data_dict = self.convert_to_serializable(data_dict)

        # Save the JSON-serializable data_dict to a file
        try:
            with open(myPathName, 'w') as json_file:
                json.dump(serializable_data_dict, json_file, indent=4)
            # print(f"File saved successfully at {myPathName}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

        
    #--------------------------------------------------------
    def getFormAndSave(self, K, audio_path, id_file, path):
        #first get the audio data
        self.getData(audio_path)
        
        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=self.y, sr=self.sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)), ref=np.max)
        
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)

        # For plotting purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0), sr=self.sr)
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))
        
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
        
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path
        
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k

        k = K
        X = evecs[:, :k] / Cnorm[:, k-1:k]
        
        # Check for NaN or infinite values in X and clean them
        if np.isnan(X).any() or np.isinf(X).any():
            X[np.isnan(X)] = 0.0
            X[np.isinf(X)] = 0.0
        
        n_init = 10
        print(f'K: {k} - n_init: {n_init}')
        KM = KMeans(n_clusters=k, n_init=n_init)

        seg_ids = KM.fit_predict(X)
        
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segments = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)
        
        #Identify unique labels in the order they first appear
        unique_labels = []
        for label in bound_segments:
            if label not in unique_labels:
                unique_labels.append(label)

        #Create a mapping from old labels to new labels starting from 1
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        #Apply the mapping to bound_segments to generate new_bound_segments
        new_bound_segments = [label_mapping[label] for label in bound_segments]
        
        #Populate the data dictionary    
        self.data_dict['sr'] = self.sr
        self.data_dict['chords'] = self.chords
        self.data_dict['bars'] = self.bars
        self.data_dict['bound_frames'] = bound_frames
        self.data_dict['bound_segments'] = new_bound_segments
        
        self.saveData(self.data_dict, id_file, path)
        
        return self.data_dict

    #--------------------------------------------------------
    def align_bars_with_sections(self, bars, section_boundaries, tempo=None):
        """
        Takes existing bars and shifts them to better align with section boundaries.
        This function is designed to work directly with formExtractor's bar format.
        
        Parameters:
        -----------
        bars : list of list
            List of bars where each bar is a list of beat times
            Format: [[bar1_beat1, bar1_beat2, bar1_beat3, bar1_beat4], [bar2_beat1, ...], ...]
        section_boundaries : list or np.ndarray
            Time points (in seconds) where sections change
        tempo : float, optional
            Tempo in BPM. If None, it will be estimated from the bars
            
        Returns:
        --------
        list of list
            Aligned bars in the same format as input
        """
        
        
        # 1. Extract the first beat of each bar to get bar start times
        bar_start_times = [bar[0] for bar in bars if len(bar) > 0]
        
        # 2. Calculate bar duration and beats per bar
        if len(bars) >= 2 and len(bars[0]) >= 2 and len(bars[1]) >= 1:
            beats_per_bar = len(bars[0])  # Typically 4 for 4/4 time
            bar_duration = bars[1][0] - bars[0][0]  # Duration from start of bar 1 to start of bar 2
            beat_duration = bar_duration / beats_per_bar
        else:
            # Default values if we can't calculate from bars
            if tempo is None:
                tempo = 120.0  # Default tempo if not provided
            beats_per_bar = 4  # Assume 4/4 time
            beat_duration = 60.0 / tempo
            bar_duration = beat_duration * beats_per_bar
        
        print(f"Bar analysis: {beats_per_bar} beats per bar, {bar_duration:.3f}s per bar, {beat_duration:.3f}s per beat")
        
        # 3. Try different offsets (0 to beats_per_bar-1 beats) and score each
        possible_offsets = np.arange(beats_per_bar) * beat_duration
        offset_scores = np.zeros(len(possible_offsets))
        
        for i, offset in enumerate(possible_offsets):
            # Apply this offset to bar times
            shifted_bar_times = np.array(bar_start_times) + offset
            
            # Score how well these aligned bars match section boundaries
            score = 0
            for boundary in section_boundaries:
                # Find distance to closest bar
                distances = np.abs(shifted_bar_times - boundary)
                min_distance = np.min(distances)
                
                # Score is higher when distance is smaller (exponential decay)
                score += np.exp(-min_distance / (0.2 * bar_duration))
            
            offset_scores[i] = score
        
        # 4. Find the best offset
        best_offset_idx = np.argmax(offset_scores)
        best_offset = possible_offsets[best_offset_idx]
        
        print(f"Offset analysis: tested {len(possible_offsets)} offsets")
        print(f"Best offset: {best_offset:.3f}s (offset {best_offset_idx} of {beats_per_bar})")
        print(f"Offset scores: {offset_scores}")
        
        # 5. Apply the best offset to all bars
        aligned_bars = []
        for bar in bars:
            if len(bar) > 0:
                # Shift each beat in the bar by the offset
                aligned_bar = [beat + best_offset for beat in bar]
                aligned_bars.append(aligned_bar)
        
        return aligned_bars

    #--------------------------------------------------------
    def optimize_song_structure(self, song, chords, beats_reference, K=4, min_duration=3.5, plot_it=False):
        """
        Performs the complete analysis pipeline:
        1. Load audio data 
        2. Extract form structure
        3. Align bars with section boundaries
        4. Create data dictionary with optimized structure
        
        Parameters:
        -----------
        song : str
            Path to the audio file
        K : int, optional
            Number of clusters for structure analysis
        min_duration : float, optional
            Minimum duration of segments in seconds
            
        Returns:
        --------
        dict
            Data dictionary with the optimized structure
        """

        # 1. Get audio data
        self.getData(song)
        
        # 2. Perform structural analysis
        C = self.amplitud_to_db(self.y, self.sr, False)
        Csync, beats, beat_times = self.sync(self.y, self.sr, C)
        
        # 3. Extract segmentation using Laplacian method
        bound_frames, bound_segments = self.laplacian_2(
            self.y, self.sr, C, Csync, beats, beat_times, 
            K=K, plot_it=plot_it, min_duration=min_duration
        )
        
        # 4. Convert bound_frames to time for alignment
        bound_times = librosa.frames_to_time(bound_frames, sr=self.sr)
        
        # 5. asign chords
        self.chords = chords

        # 6. Create data dictionary with aligned structure
        data_dict = self.populateDict(
            self.sr,
            self.chords,
            beats_reference,
            bound_times,
            bound_frames,
            bound_segments,
        )
        
        return data_dict