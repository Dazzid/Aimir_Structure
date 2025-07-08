import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from IPython.display import Audio, display
from matplotlib import colors as mcolors
import re


#--------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import librosa.display
import numpy as np

my_colors = [
    "#6BC1FF",
    "#FFD269",
    "#324DFF",
    "#FFB700",
    "#08F7FF",
    "#FFAA00",
    "#FFCC4D",
    "#FFE091",
    "#FFFF00",
    "#FFF388" 
]

def plotWaveform(y, sr, chords, beats, bound_frames, new_bound_segs, size_x, size_y, 
                 colormap_name='viridis', beat_offset=0.0, diagnostic=False, custom_colors=my_colors):
    """
    Plot a waveform with beats, section boundaries, and chord labels on the x-axis.
    """
    # Handle color mapping
    if custom_colors is None:
        colormap = plt.get_cmap(colormap_name)
        num_colors = len(set(new_bound_segs))
        colors = colormap(np.linspace(0, 1, num_colors))
    else:
        colors = custom_colors
    
    # Calculate the total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Create the figure and axis
    if diagnostic:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(size_x, size_y), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 1]})
        ax = ax1
    else:
        fig, ax = plt.subplots(figsize=(size_x, size_y))
    
    # Plot the waveform without downsampling
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#000000", alpha=0.5)
    
    # Apply beat offset if provided
    adjusted_beats = beats + beat_offset
    
    # Convert bound frames to times
    bound_times = librosa.frames_to_time(bound_frames, sr=sr)
    
    # Plot section boundaries with custom colors
    for i, (start, label) in enumerate(zip(bound_times, new_bound_segs)):
        # Calculate the end time for this section
        if i < len(bound_times) - 1:
            end = bound_times[i + 1]
        else:
            end = total_duration
        
        # Ensure label index is in range for the colors array
        color_idx = (label - 1) % len(colors)
        rect = patches.Rectangle(
            (start, ax.get_ylim()[0]),
            end - start,
            ax.get_ylim()[1] - ax.get_ylim()[0],
            facecolor=colors[color_idx],
            alpha=0.5
        )
        ax.add_patch(rect)
        
        # Add section label text at the midpoint of the section
        midpoint = (start + end) / 2
        ax.text(
            midpoint,
            0,
            f"Section {label}",
            rotation=90,
            verticalalignment='center',
            fontsize=16,
            color='black',
            weight='normal'
        )
    
    # Plot beat visualization
    for i, beat_time in enumerate(adjusted_beats):
        # Draw vertical line for each beat
        ax.axvline(x=beat_time, color="#C5C5C5", linestyle='-', linewidth=0.7, alpha=0.6)
    
    # For diagnostic view, add spectrogram to bottom plot
    if diagnostic:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', sr=sr, x_axis='time', ax=ax2)
        
        for beat_time in adjusted_beats:
            ax2.axvline(x=beat_time, color='#990000', linestyle='-', linewidth=0.7, alpha=0.6)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        for onset_time in onset_times:
            ax2.axvline(x=onset_time, color='green', linestyle='--', linewidth=0.7, alpha=0.6)
        
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (mm:ss)')
    
    # Set x-axis limits
    ax.set_xlim(0, total_duration)
    
    # Replace time ticks with chord ticks
    if chords and len(chords) > 0:
        # Clear existing ticks and their labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Place ticks at chord positions
        chord_positions = [chord['beat_start'] for chord in chords]
        chord_labels = [chord['functional_harmony']['functional'] for chord in chords]
        
        # Set new ticks and their labels
        ax.set_xticks(chord_positions)
        ax.set_xticklabels(chord_labels, rotation=90, fontsize=10)
        
        # Make sure tick labels are visible
        for tick in ax.get_xticklabels():
            tick.set_verticalalignment('top')
    
    # Set axis labels and title
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('')  # No x-label needed since chords are the labels
    title = 'Audio Waveform with Structural Analysis'
    if beat_offset != 0:
        title += f' (Beat Offset: {beat_offset:.3f}s)'
    ax.set_title(title)
    
    # Display diagnostic information if requested
    if diagnostic:
        avg_beat_interval = np.mean(np.diff(adjusted_beats)) if len(adjusted_beats) > 1 else 0
        tempo = 60 / avg_beat_interval if avg_beat_interval > 0 else 0
        
        diagnostic_text = (
            f"Diagnostics:\n"
            f"• Audio duration: {total_duration:.2f}s\n"
            f"• Sample rate: {sr} Hz\n"
            f"• Beat count: {len(beats)}\n"
            f"• Avg beat interval: {avg_beat_interval:.3f}s\n"
            f"• Estimated tempo: {tempo:.1f} BPM\n"
            f"• Beat offset applied: {beat_offset:.3f}s"
        )
        
        fig.text(0.98, 0.98, diagnostic_text, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))
    
    # Add extra space at bottom for the rotated chord labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()
    
    return fig

#--------------------------------------------------------------------------------
def plotChordsBars(chords, bars, bound_frames, bound_segs, size_x=40, size_y=5):
    # Choose a colormap from Matplotlib
    section_colormap_name = 'rainbow'  # You can change this to any colormap name, e.g., 'tab20', 'viridis', 'Blues', etc.

    # Choose colormaps from Matplotlib
    section_colormap = plt.get_cmap(section_colormap_name)
    num_colors = len(set(bound_segs))
    custom_colors = section_colormap(np.linspace(0, 1, num_colors))

    # Convert bound frames to times
    bound_times = librosa.frames_to_time(bound_frames)

    # Plot the waveform
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    
    # Plot section boundaries with custom colors
    for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
        color_idx = label % len(custom_colors)  # Ensure we don't exceed the color map length
        rect = patches.Rectangle((interval[0], plt.ylim()[0]), interval[1] - interval[0], plt.ylim()[1] - plt.ylim()[0], facecolor=custom_colors[color_idx], alpha=0.35)
        ax.add_patch(rect)
        # Add section label text at the midpoint of the section
        midpoint = (interval[0] + interval[1]) / 2
        plt.text(midpoint, 0.98, f"Section {label}", rotation=90, verticalalignment='top', fontsize=12, color='black', weight='normal')

    # Plot chord changes and annotate chords
    for chord in chords[1:-1]:
        #plt.axvline(x=chord.timestamp, color='r', linestyle='--', linewidth=0.5)
        plt.text(chord.timestamp, 0.5, chord.chord, rotation=90, verticalalignment='center', fontsize=10, color='black', weight='normal')

    # Plot vertical lines for each bar
    for bar in bars:
        plt.axvline(x=bar[0], color='#555555', linestyle='dotted', linewidth=0.5)
        plt.text(bar[0] + 0.01, plt.ylim()[0], f"Bar {bars.index(bar) + 1}", rotation=90, verticalalignment='bottom', fontsize=10, color='#000')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
#--------------------------------------------------------------------------------
def plotSpectrogram(sr, C, chords, bars, bound_frames, new_bound_segs, BINS_PER_OCTAVE, size_x, size_y):
    # Choose colormaps from Matplotlib
    section_colormap_name = 'Blues'
    spectrogram_colormap_name = 'Blues'

    # Generate a list of colors using the chosen colormap for sections
    section_colormap = plt.get_cmap(section_colormap_name)
    num_colors = len(set(new_bound_segs))
    custom_colors = section_colormap(np.linspace(0, 1, num_colors))

    # Generate the spectrogram colormap
    spectrogram_colormap = plt.get_cmap(spectrogram_colormap_name)

    # Generate the spectrogram
    bound_times = librosa.frames_to_time(bound_frames)
    freqs = librosa.cqt_frequencies(n_bins=C.shape[0], fmin=librosa.note_to_hz('C1'), bins_per_octave=BINS_PER_OCTAVE)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    librosa.display.specshow(C, y_axis='cqt_hz', sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', ax=ax, cmap=spectrogram_colormap)

    # Set the y-axis limit to ensure the maximum frequency displayed is 4096 Hz
    ax.set_ylim([freqs[0], 4096])

    # Plot section boundaries with custom colors
    for interval, label in zip(zip(bound_times, bound_times[1:]), new_bound_segs):
        color_idx = label % len(custom_colors)  # Ensure we don't exceed the color map length
        ax.add_patch(patches.Rectangle((interval[0], freqs[0]), interval[1] - interval[0], 4096 - freqs[0], facecolor=custom_colors[color_idx], alpha=0.5))

    # Plot chord changes and annotate chords
    for chord in chords[1:-1]:
        plt.text(chord.timestamp, freqs[-1]+500, chord.chord, rotation=90, verticalalignment='bottom', fontsize=13, color='black')

    # Plot vertical lines for each bar
    for bar in bars:
        plt.axvline(x=bar[0], color='#555555', linestyle='dotted', linewidth=0.5)
        plt.text(bar[0]+0.01, freqs[0], f"Bar {bars.index(bar) + 1}", rotation=90, verticalalignment='bottom', fontsize=12, color='#000')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

#--------------------------------------------------------------------------------
def plotSections(y, sr, chords, bars, bound_frames, new_bound_segs, size_x, size_y, bars_per_subplot=32):
    # Choose a colormap from Matplotlib
    colormap_name = 'Blues'
    colormap = plt.get_cmap(colormap_name)
    num_colors = len(set(new_bound_segs))
    custom_colors = colormap(np.linspace(0, 1, num_colors))

    # Convert bound frames to times
    bound_times = librosa.frames_to_time(bound_frames)

    # Function to plot a section
    def plot_section(ax, y_section, sr, section_start, section_end, chords, bars, new_bound_segs, bound_times, custom_colors):
        # Plot the waveform for the current section
        librosa.display.waveshow(y_section, sr=sr, ax=ax, x_axis='time', color= '#00AAFF', alpha=0.9)

        # Plot section boundaries with custom colors
        for interval, label in zip(zip(bound_times, bound_times[1:]), new_bound_segs):
            if interval[1] < section_start or interval[0] > section_end:
                continue
            color_idx = label % len(custom_colors)
            rect = patches.Rectangle((interval[0] - section_start, ax.get_ylim()[0]), interval[1] - interval[0], ax.get_ylim()[1] - ax.get_ylim()[0], facecolor=custom_colors[color_idx], alpha=0.5)
            ax.add_patch(rect)
            # Add section label text at the midpoint of the section
            midpoint = (interval[0] + interval[1]) / 2 - section_start
            ax.text(midpoint, 0, f"Section {label}", rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=16, color='black', weight='normal')

        # Plot chord changes and annotate chords
        for chord in chords[1:-1]:
            if chord.timestamp < section_start or chord.timestamp > section_end:
                continue
            ax.text(chord.timestamp - section_start, ax.get_ylim()[1] + 0.01, chord.chord, rotation=90, verticalalignment='bottom', fontsize=12, color='black', weight='light')

        # Plot vertical lines for each bar
        for bar in bars:
            if bar[0] < section_start or bar[0] > section_end:
                continue
            ax.axvline(x=bar[0] - section_start, color='#555555', linestyle='dotted', linewidth=0.5)
            ax.text(bar[0] - section_start + 0.01, ax.get_ylim()[0], f"Bar {bars.index(bar) + 1}", rotation=90, verticalalignment='bottom', fontsize=12, color='#000')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

    # Determine the number of subplots needed
    num_subplots = int(np.ceil(len(bars) / bars_per_subplot))

    fig, axs = plt.subplots(num_subplots, 1, figsize=(size_x, size_y * num_subplots), sharex=False)

    if num_subplots == 1:
        axs = [axs]  # Ensure axs is iterable

    # Plot each section
    for subplot_idx, ax in enumerate(axs):
        start_bar = subplot_idx * bars_per_subplot
        end_bar = min((subplot_idx + 1) * bars_per_subplot, len(bars))
        start_time = bars[start_bar][0]
        if end_bar == len(bars):
            end_time = librosa.get_duration(y=y, sr=sr)  # Ensure we cover to the end of the track
        else:
            end_time = bars[end_bar - 1][-1]
        
        # Extract the segment of the waveform
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        y_section = y[start_sample:end_sample]
        
        # Plot the section
        plot_section(ax, y_section, sr, start_time, end_time, chords, bars, new_bound_segs, bound_times, custom_colors)

    plt.tight_layout()
    plt.show()

    # Generate audio segments for each section
    for subplot_idx in range(num_subplots):
        start_bar = subplot_idx * bars_per_subplot
        end_bar = min((subplot_idx + 1) * bars_per_subplot, len(bars))
        start_time = bars[start_bar][0]
        if end_bar == len(bars):
            end_time = librosa.get_duration(y=y, sr=sr)  # Ensure we cover to the end of the track
        else:
            end_time = bars[end_bar - 1][-1]
        
        # Extract the segment of the waveform
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        y_section = y[start_sample:end_sample]
        
        # Display the audio segment
        print(f"Audio for section {subplot_idx + 1}")
        display(Audio(data=y_section, rate=sr))
        
#-----------------------------------------------------------
#Plot only the chords
def plotChords(y, sr, chords, size_x=40, size_y=5):
    # Create the plot
    plt.figure(figsize=(size_x, size_y))
    librosa.display.waveshow(y, sr=sr, color='#FFAA00')

    font = {'family' : 'Helvetica',
            'weight' : 'light',
            'size'   : 12}
    plt.rc('font', **font)

    # Plot vertical lines for each chord change
    for chord in chords[1:-1]:
        plt.axvline(x=chord.timestamp, color='#777', linestyle='dotted', linewidth=0.5)
        plt.text(chord.timestamp, max(y)+0.15, chord.chord, rotation=90, verticalalignment='bottom', color='black')

    # Display the plot
    plt.show()
    
#-----------------------------------------------------------
def relabel_54_to_sus(func_string):
    """
    Check if music21 has added 54 or 52 which are sus chords
    """
    if '54' in func_string:
        # Remove '54', take left part, uppercase, and append '_sus'
        base_func = func_string.replace('54', '')
        base_func = base_func.upper()
        return f"{base_func} sus"
    elif '52' in func_string:
        # Remove '52', take left part, uppercase, and append '_sus'
        base_func = func_string.replace('52', '')
        base_func = base_func.upper()
        return f"{base_func} sus"
    else:
        return func_string
    
#-----------------------------------------------------------
def plot_functional_harmony_map(chords_data, beats_per_row=16):
    """
    Plot a functional harmony map visualization with EXACT same alignment approach as plot_chord_score_map.
    Uses integer beat indices (beat_idx and beat_idx_end) instead of floating point beat timings.
    
    Args:
        chords_data: List of chord dictionaries with functional_harmony data
        beats_per_row: Number of beats to display per row in the visualization
        
    Returns:
        Matplotlib figure object with the visualization
    """
    
    # Function to determine if text should be white or black based on background color
    def get_text_color(bg_color):
        """
        Returns white or black text color depending on background darkness
        """
        # If bg_color is a hex string, convert to RGB
        if isinstance(bg_color, str) and bg_color.startswith('#'):
            # Convert hex to RGB
            bg_color = bg_color.lstrip('#')
            r, g, b = tuple(int(bg_color[i:i+2], 16)/255 for i in (0, 2, 4))
        else:
            r, g, b = bg_color
            
        # Calculate luminance - standard formula for perceived brightness
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Return white text for dark backgrounds, black for light backgrounds
        return "white" if luminance < 0.5 else "black"
    
    # If we receive a dict with 'chords' key, extract the chords list (same as plot_chord_score_map)
    if isinstance(chords_data, dict) and 'chords' in chords_data:
        chords = chords_data['chords']
    else:
        chords = chords_data
    
    # Create arrays of functional harmony, beat indices, and chord names - EXACTLY as plot_chord_score_map does
    chord_names = []
    beat_starts = []
    beat_ends = []
    functional_info = []
    
    for chord in chords:
        # Store the original chord name for reference
        chord_names.append(chord['chord_name'])
        
        # Use INTEGER beat indices EXACTLY like plot_chord_score_map does
        beat_starts.append(chord['beat_idx'])
        beat_ends.append(chord['beat_idx_end'] + 1)  # +1 to make it inclusive
        
        # Extract functional harmony info
        if 'functional_harmony' in chord:
            roman_numeral = chord['functional_harmony'].get('roman_numeral', '')
            func = chord['functional_harmony'].get('functional', '')
            alter = chord['functional_harmony'].get('alterations', '')
            functional_info.append(roman_numeral)

        else:
            functional_info.append('')
            
    functional_info = [relabel_54_to_sus(func) for func in functional_info]  # Apply relabeling
    
    # Calculate total beats based on the actual data - EXACTLY as plot_chord_score_map does
    total_beats = int(np.ceil(max(beat_ends)))
    n_rows = int(np.ceil(total_beats / beats_per_row))
    
    # Create figure with same dimensions as plot_chord_score_map
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(16, beats_per_row), 2.0 * n_rows), squeeze=False)
    
    # Define color mapping for Roman numerals (functional harmony categories)
    roman_colors = {
        'I': "#AEE1FF",    # Tonic - white
        'i': "#AEE1FF",    # Tonic - white
        'IV': "#FFC053",   # Subdominant - orange
        'iv': "#FFC053",   # Subdominant - orange
        'V': "#FFFFFF",    # Dominant - red-orange
        'v': "#FFFFFF",    # Dominant - red-orange 
        'II': "#45A5FF",   # Supertonic - light blue
        'ii': "#45A5FF",   # Supertonic - light blue
        'VI': "#67E1FF",   # Submediant - cyan
        'vi': "#67E1FF",   # Submediant - cyan
        'III': "#00C8FF",  # Mediant - light gray
        'iii': "#00C8FF",  # Mediant - light gray
        'VII': "#FF3C63",  # Leading tone - pink/red
        'vii': "#FF3C63",  # Leading tone - pink/red
        '': "#D9D9D9"  # Default - gray
    }
    
    # Add support for flat/sharp modifiers
    for base in list(roman_colors.keys()):
        if base not in ['']:
            roman_colors[f'b{base}'] = roman_colors[base]  # Add flat version
            roman_colors[f'#{base}'] = roman_colors[base]  # Add sharp version
    
    # Process each row - EXACTLY as plot_chord_score_map does
    for row in range(n_rows):
        ax = axes[row, 0]
        
        # Define beat range for this row - EXACTLY as plot_chord_score_map does
        row_start = row * beats_per_row
        row_end = min((row + 1) * beats_per_row, total_beats)
        
        # Draw beat grid lines - EXACTLY as plot_chord_score_map does
        for beat in range(row_start, row_end + 1):
            # Style for beat lines (stronger lines every 4 beats)
            if beat % 4 == 0:
                ax.axvline(beat - row_start, color="#D8D8D8", linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)
                # Add beat number label
                ax.text(beat - row_start, 0.1, str(beat), ha='center', fontsize=8, color='gray')
            else:
                ax.axvline(beat - row_start, color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Plot each chord in this row - EXACTLY as plot_chord_score_map does
        for func_info, chord_name, start, end in zip(functional_info, chord_names, beat_starts, beat_ends):
            # Only plot chords that overlap with this row - EXACTLY as plot_chord_score_map does
            if end <= row_start or start >= row_end:
                continue
            
            # Calculate position within the row - EXACTLY as plot_chord_score_map does
            plot_start = max(start, row_start) - row_start
            plot_end = min(end, row_end) - row_start
            width = plot_end - plot_start
            
            if width > 0:
                # Get the base function (without alterations) for color mapping
                base_func = func_info
                for alter in ['7', 'dim', 'aug', 'sus', '6', '9']:
                    if alter in func_info:
                        base_func = func_info.replace(alter, '')
                        break
                
                # If base_func is still not recognized, try just the first one or two chars
                if base_func not in roman_colors and len(base_func) > 1:
                    # Try with just the base numeral (I, V, etc)
                    if base_func[:1] in roman_colors:
                        base_func = base_func[:1]
                    # Or with flat/sharp prefix (bII, #IV, etc)
                    elif len(base_func) > 2 and base_func[:2] in roman_colors:
                        base_func = base_func[:2]
                
                # Determine color based on functional harmony
                color = roman_colors.get(base_func, "#D9D9D9")  # Default gray if not found
                
                # Draw chord rectangle - EXACTLY as plot_chord_score_map does
                ax.barh(1, width, left=plot_start, height=2.0, color=color, edgecolor="#4A4A4A", alpha=0.8)
                
                # Add functional information + chord name - adjust font size based on width
                # Increase font size by making the base size larger and scaling factor higher
                font_size = min(14, max(12, width * 4))
                
                # Determine display based on width
                if width > 4:
                    # For wide chords, show both functional info and chord name
                    display_text = f"{func_info}\n({chord_name})"
                elif width > 1.5:
                    # For medium width, just show functional info
                    display_text = func_info
                else:
                    # For narrow width, still just show functional info but rotated
                    display_text = func_info
                
                # Determine text color based on background color
                text_color = get_text_color(color)
                
                # Add text with same rotation logic as plot_chord_score_map
                ax.text(plot_start + width/2, 1, display_text, 
                       va='center', ha='center', fontsize=font_size, 
                       color=text_color, rotation=0 if width > 1.5 else 90, 
                       fontweight='normal')
        
        # Set axis properties - EXACTLY as plot_chord_score_map does
        ax.set_xlim(0, row_end - row_start)
        ax.set_ylim(0, 2)
        ax.set_yticks([])
        
        # Set custom x-ticks for clarity - EXACTLY as plot_chord_score_map does
        tick_interval = 2 if beats_per_row > 16 else 1
        ax.set_xticks(np.arange(0, row_end - row_start + 1, tick_interval))
        ax.set_xticklabels([int(row_start + t) for t in ax.get_xticks()])
        
        # Add row labels - similar to plot_chord_score_map, adapted for functional harmony
        ax.set_xlabel(f'Beats', fontsize=12)
        ax.set_title(f'Functional Harmony Map (Beats {row_start} - {row_end-1})', fontsize=14)
        
        # Add a light grid for better readability - EXACTLY as plot_chord_score_map does
        ax.grid(True, axis='x', alpha=0.2, linestyle='-')
    
    # Create comprehensive legend for all used functional categories
    legend_handles = []
    legend_labels = []
    
    # Define the order we want the legend items to appear
    legend_order = [
        ('Tonic (I/i)', ['I', 'i']),
        ('Supertonic (II/ii)', ['II', 'ii']),
        ('Mediant (III/iii)', ['III', 'iii']),
        ('Subdominant (IV/iv)', ['IV', 'iv']),
        ('Dominant (V/v)', ['V', 'v']),
        ('Submediant (VI/vi)', ['VI', 'vi']),
        ('Leading Tone (VII/vii)', ['VII', 'vii']),
        ('Unknown', ['Unknown'])
    ]
    
    # Create a set of all used functional harmony base types
    used_funcs = set()
    for func in functional_info:
        # Strip alterations to get base function
        base = func
        for alter in ['7', 'dim', 'aug', 'sus', '6', '9']:
            if alter in func:
                base = func.replace(alter, '')
        
        # Add the base function to used set
        if base in roman_colors:
            used_funcs.add(base)
        elif base[:1] in roman_colors:  # Try just first character
            used_funcs.add(base[:1])
        elif len(base) > 1 and base[:2] in roman_colors:  # Try first two chars for b/# prefixes
            used_funcs.add(base[:2])
    
    # Add legend entries in our preferred order
    for group_name, funcs in legend_order:
        if any(f in used_funcs for f in funcs):
            color = roman_colors[funcs[0]]
            # Create rectangle for legend
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
            legend_labels.append(group_name)
    
    # Place legend below the last subplot with better spacing and visibility
    if legend_handles:
        legend = axes[-1, 0].legend(
            legend_handles, 
            legend_labels, 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(4, len(legend_handles)),  # Adjust columns based on number of items
            frameon=True, 
            fontsize=12  # Increase legend font size
        )
        # Set edgecolor for legend items to improve visibility
        for handle in legend.legend_handles if hasattr(legend, 'legend_handles') else legend.legendHandles:
            handle.set_edgecolor('black')
    
    plt.tight_layout()
    return fig


#--------------------------------------------------------------------------------
def plot_chord_score_map(chords_data, beats_per_row=16):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # If we receive a dict with 'chords' key, extract the chords list
    if isinstance(chords_data, dict) and 'chords' in chords_data:
        chords = chords_data['chords']
    else:
        chords = chords_data
    
    # Create arrays of chord names, beat indices, and timings
    chord_names = []
    beat_starts = []
    beat_ends = []
    
    for chord in chords:
        chord_names.append(chord['chord_name'])
        beat_starts.append(chord['beat_idx'])  # Using the integer beat index
        beat_ends.append(chord['beat_idx_end'] + 1)  # +1 to make it inclusive
    
    # Calculate total beats based on the actual data
    total_beats = int(np.ceil(max(beat_ends)))
    n_rows = int(np.ceil(total_beats / beats_per_row))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(16, beats_per_row), 2.0 * n_rows), squeeze=False)
    
    for row in range(n_rows):
        ax = axes[row, 0]
        
        # Define beat range for this row
        row_start = row * beats_per_row
        row_end = min((row + 1) * beats_per_row, total_beats)
        
        # Draw beat grid lines
        for beat in range(row_start, row_end + 1):
            # Style for beat lines (stronger lines every 4 beats)
            if beat % 4 == 0:
                ax.axvline(beat - row_start, color="#D8D8D8", linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)
                # Add beat number label
                ax.text(beat - row_start, 0.1, str(beat), ha='center', fontsize=8, color='black')
            else:
                ax.axvline(beat - row_start, color='black', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Plot each chord in this row
        for chord_name, start, end in zip(chord_names, beat_starts, beat_ends):
            # Only plot chords that overlap with this row
            if end <= row_start or start >= row_end:
                continue
            
            # Calculate position within the row
            plot_start = max(start, row_start) - row_start
            plot_end = min(end, row_end) - row_start
            width = plot_end - plot_start
            
            if width > 0:
                # Plot the chord box - use different colors for different chord types
                color = "#51D1FF"  # Default color
                if re.search(r'm7b5\b', chord_name):
                    color = "#FA884B"
                    
                elif re.search(r'mmaj7\b', chord_name):  # minor-major seventh
                    color = "#5DD4FF"
                    
                elif re.search(r'maj7\b', chord_name):
                    color = "#FFA406"
                
                elif re.search(r'm(7|9|11|13)?\b', chord_name):  # 'm', 'm7', etc
                    color = "#83C9FF"
                    
                elif re.search(r'7\b', chord_name):
                    color = '#FFFFFF'
                    
                elif re.search(r'sus(4)?\b', chord_name):
                    color = "#E8E8E8"
                    
                
                # Draw chord rectangle
                ax.barh(1, width, left=plot_start, height=2.0, 
                        color=color, edgecolor="#525252", alpha=0.8)
                
                # Add chord name - adjust font size based on width
                font_size = min(14, max(12, width * 3))
                ax.text(plot_start + width/2, 1, chord_name, 
                        va='center', ha='center', fontsize=font_size, 
                        color="#000000", rotation=0 if width > 1.5 else 90, 
                        fontweight='normal')
        
        # Set axis properties
        ax.set_xlim(0, row_end - row_start)
        ax.set_ylim(0, 2)
        ax.set_yticks([])
        
        # Set custom x-ticks for clarity
        tick_interval = 2 if beats_per_row > 16 else 1
        ax.set_xticks(np.arange(0, row_end - row_start + 1, tick_interval))
        ax.set_xticklabels([int(row_start + t) for t in ax.get_xticks()])
        
        # Add row labels
        ax.set_xlabel(f'Beats')
        ax.set_title(f'Chord Map (Beats {row_start} - {row_end-1})')
        
        # Add a light grid for better readability
        ax.grid(True, axis='x', alpha=0.2, linestyle='-')
    
    plt.tight_layout()
    return fig


def plot_structure_map(structure_data, beats_per_row=16):
    """
    Plot a structure map visualization showing segment boundaries and labels.
    Uses beat-based alignment similar to plot_chord_score_map.
    
    Args:
        structure_data: Dictionary containing 'bound_frames', 'bound_segments', 'beats', 'sr'
        beats_per_row: Number of beats to display per row in the visualization
        
    Returns:
        Matplotlib figure object with the visualization
    """
    
    # Extract structure data
    bound_frames = structure_data['bound_frames']
    bound_segments = structure_data['bound_segments']
    beats = structure_data['beats']
    sr = structure_data['sr']
    
    # Convert bound_frames to beat indices
    # Find which beat each boundary frame corresponds to
    bound_beat_indices = []
    for frame in bound_frames:
        # Convert frame to time
        frame_time = librosa.frames_to_time(frame, sr=sr)
        # Find closest beat
        beat_diffs = np.abs(np.array(beats) - frame_time)
        closest_beat_idx = np.argmin(beat_diffs)
        bound_beat_indices.append(closest_beat_idx)
    
    # Calculate total beats
    total_beats = len(beats)
    n_rows = int(np.ceil(total_beats / beats_per_row))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(16, beats_per_row), 2.0 * n_rows), squeeze=False)
    
    # Define colors for different segments
    segment_colors = {
        1: "#AEE1FF",   # Light blue
        2: "#FFC053",   # Orange
        3: "#67E1FF",   # Cyan
        4: "#FF3C63",   # Pink/red
        5: "#45A5FF",   # Blue
        6: "#00C8FF",   # Darker cyan
        7: "#FFD269",   # Yellow
        8: "#FFFFFF",   # White
    }
    
    for row in range(n_rows):
        ax = axes[row, 0]
        
        # Define beat range for this row
        row_start = row * beats_per_row
        row_end = min((row + 1) * beats_per_row, total_beats)
        
        # Draw beat grid lines
        for beat in range(row_start, row_end + 1):
            if beat % 4 == 0:
                ax.axvline(beat - row_start, color="#D8D8D8", linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)
                # Add beat number label
                ax.text(beat - row_start, 0.1, str(beat), ha='center', fontsize=8, color='gray')
            else:
                ax.axvline(beat - row_start, color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Plot each segment in this row
        for i in range(len(bound_beat_indices)):
            segment_start_beat = bound_beat_indices[i]
            segment_label = bound_segments[i]
            
            # Determine segment end
            if i < len(bound_beat_indices) - 1:
                segment_end_beat = bound_beat_indices[i + 1]
            else:
                segment_end_beat = total_beats
            
            # Only plot segments that overlap with this row
            if segment_end_beat <= row_start or segment_start_beat >= row_end:
                continue
            
            # Calculate position within the row
            plot_start = max(segment_start_beat, row_start) - row_start
            plot_end = min(segment_end_beat, row_end) - row_start
            width = plot_end - plot_start
            
            if width > 0:
                # Get color for this segment
                color = segment_colors.get(segment_label, "#D9D9D9")  # Default gray
                
                # Draw segment rectangle
                ax.barh(1, width, left=plot_start, height=2.0, 
                        color=color, edgecolor="#4A4A4A", alpha=0.8)
                
                # Add segment label - adjust font size based on width
                font_size = min(16, max(12, width * 4))
                segment_text = f"Section {segment_label}"
                
                # Determine text color based on background
                if color in ["#FFFFFF", "#AEE1FF", "#FFC053", "#FFD269"]:
                    text_color = "black"
                else:
                    text_color = "white"
                
                ax.text(plot_start + width/2, 1, segment_text, 
                        va='center', ha='center', fontsize=font_size, 
                        color=text_color, rotation=0 if width > 2 else 90, 
                        fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(0, row_end - row_start)
        ax.set_ylim(0, 2)
        ax.set_yticks([])
        
        # Set custom x-ticks for clarity
        tick_interval = 4 if beats_per_row > 16 else 2
        ax.set_xticks(np.arange(0, row_end - row_start + 1, tick_interval))
        ax.set_xticklabels([int(row_start + t) for t in ax.get_xticks()])
        
        # Add row labels
        ax.set_xlabel(f'Beats', fontsize=12)
        ax.set_title(f'Structure Map (Beats {row_start} - {row_end-1})', fontsize=14)
        
        # Add a light grid for better readability
        ax.grid(True, axis='x', alpha=0.2, linestyle='-')
    
    # Create legend for segment colors
    legend_handles = []
    legend_labels = []
    unique_segments = sorted(set(bound_segments))
    
    for segment in unique_segments:
        color = segment_colors.get(segment, "#D9D9D9")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
        legend_labels.append(f'Section {segment}')
    
    # Place legend below the last subplot
    if legend_handles:
        axes[-1, 0].legend(
            legend_handles, 
            legend_labels, 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(4, len(legend_handles)),
            frameon=True, 
            fontsize=12
        )
    
    plt.tight_layout()
    return fig

def plot_combined_chord_structure_map(chords_data, structure_data, beats_per_row=16):
    """
    Plot a combined visualization showing both chords and structure segments.
    Chords are displayed as before, with a colored structure bar at the top.
    
    Args:
        chords_data: List of chord dictionaries or dict with 'chords' key
        structure_data: Dictionary containing 'bound_frames', 'bound_segments', 'beats', 'sr'
        beats_per_row: Number of beats to display per row in the visualization
        
    Returns:
        Matplotlib figure object with the visualization
    """
    # Extract chord data
    if isinstance(chords_data, dict) and 'chords' in chords_data:
        chords = chords_data['chords']
    else:
        chords = chords_data
    
    # Extract structure data
    bound_frames = structure_data['bound_frames']
    bound_segments = structure_data['bound_segments']
    beats = structure_data['beats']
    sr = structure_data['sr']
    
    # Process chord data
    chord_names = []
    beat_starts = []
    beat_ends = []
    
    for chord in chords:
        chord_names.append(chord['chord_name'])
        beat_starts.append(chord['beat_idx'])
        beat_ends.append(chord['beat_idx_end'] + 1)  # +1 to make it inclusive
    
    # Convert bound_frames to beat indices
    bound_beat_indices = []
    for frame in bound_frames:
        frame_time = librosa.frames_to_time(frame, sr=sr)
        beat_diffs = np.abs(np.array(beats) - frame_time)
        closest_beat_idx = np.argmin(beat_diffs)
        bound_beat_indices.append(closest_beat_idx)
    
    # Calculate total beats
    total_beats = int(np.ceil(max(beat_ends)))
    n_rows = int(np.ceil(total_beats / beats_per_row))
    
    # Create figure with extra height for structure bar
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(16, beats_per_row), 2.5 * n_rows), squeeze=False)
    
    # Define colors for structure segments
    segment_colors = {
        1: "#AEE1FF",   # Light blue
        2: "#FFC053",   # Orange
        3: "#67E1FF",   # Cyan
        4: "#FF3C63",   # Pink/red
        5: "#45A5FF",   # Blue
        6: "#00C8FF",   # Darker cyan
        7: "#FFD269",   # Yellow
        8: "#FFFFFF",   # White
    }
    
    for row in range(n_rows):
        ax = axes[row, 0]
        
        # Define beat range for this row
        row_start = row * beats_per_row
        row_end = min((row + 1) * beats_per_row, total_beats)
        
        # Draw beat grid lines
        for beat in range(row_start, row_end + 1):
            if beat % 4 == 0:
                ax.axvline(beat - row_start, color="#D8D8D8", linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)
                ax.text(beat - row_start, 0.1, str(beat), ha='center', fontsize=8, color='black')
            else:
                ax.axvline(beat - row_start, color='black', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Plot structure segments at the top (y=2.2 to 2.8)
        for i in range(len(bound_beat_indices)):
            segment_start_beat = bound_beat_indices[i]
            segment_label = bound_segments[i]
            
            # Determine segment end
            if i < len(bound_beat_indices) - 1:
                segment_end_beat = bound_beat_indices[i + 1]
            else:
                segment_end_beat = len(beats)
            
            # Only plot segments that overlap with this row
            if segment_end_beat <= row_start or segment_start_beat >= row_end:
                continue
            
            # Calculate position within the row
            plot_start = max(segment_start_beat, row_start) - row_start
            plot_end = min(segment_end_beat, row_end) - row_start
            width = plot_end - plot_start
            
            if width > 0:
                color = segment_colors.get(segment_label, "#D9D9D9")
                
                # Draw structure bar at top
                ax.barh(2.5, width, left=plot_start, height=0.6, 
                        color=color, edgecolor="#4A4A4A", alpha=0.9)
                
                # Add section label if wide enough
                if width > 2:
                    text_color = "black" if color in ["#FFFFFF", "#AEE1FF", "#FFC053", "#FFD269"] else "white"
                    ax.text(plot_start + width/2, 2.5, f"Sec {segment_label}", 
                            va='center', ha='center', fontsize=10, 
                            color=text_color, fontweight='bold')
        
        # Plot chords in the main area (y=1)
        for chord_name, start, end in zip(chord_names, beat_starts, beat_ends):
            # Only plot chords that overlap with this row
            if end <= row_start or start >= row_end:
                continue
            
            # Calculate position within the row
            plot_start = max(start, row_start) - row_start
            plot_end = min(end, row_end) - row_start
            width = plot_end - plot_start
            
            if width > 0:
                # Chord colors (same as original)
                if 'maj7' in chord_name:
                    color = "#FFA406"
                elif 'm' in chord_name:
                    color = "#51D1FF"
                elif '7' in chord_name:
                    color = '#FFFFFF'
                elif 'sus' in chord_name:
                    color = "#FF4564"
                else:
                    color = "#D9D9D9"
                
                # Draw chord rectangle
                ax.barh(1, width, left=plot_start, height=2.0, 
                        color=color, edgecolor="#525252", alpha=0.8)
                
                # Add chord name
                font_size = min(14, max(12, width * 3))
                ax.text(plot_start + width/2, 1, chord_name, 
                        va='center', ha='center', fontsize=font_size, 
                        color="#000000", rotation=0 if width > 1.5 else 90, 
                        fontweight='normal')
        
        # Set axis properties
        ax.set_xlim(0, row_end - row_start)
        ax.set_ylim(0, 3.2)  # Extended to accommodate structure bar
        ax.set_yticks([])
        
        # Set custom x-ticks
        tick_interval = 2 if beats_per_row > 16 else 1
        ax.set_xticks(np.arange(0, row_end - row_start + 1, tick_interval))
        ax.set_xticklabels([int(row_start + t) for t in ax.get_xticks()])
        
        # Add labels
        ax.set_xlabel(f'Beats')
        ax.set_title(f'Combined Chord & Structure Map (Beats {row_start} - {row_end-1})')
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.2, linestyle='-')
        
        # Add y-axis labels for clarity
        ax.text(-0.5, 1, 'Chords', rotation=90, va='center', ha='right', fontsize=10, fontweight='bold')
        ax.text(-0.5, 2.5, 'Structure', rotation=90, va='center', ha='right', fontsize=10, fontweight='bold')
    
    # Create legends
    # Chord legend
    chord_legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#FFA406", edgecolor='black', label='Major 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor="#51D1FF", edgecolor='black', label='Minor'),
        plt.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", edgecolor='black', label='7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor="#FF4564", edgecolor='black', label='Sus'),
        plt.Rectangle((0, 0), 1, 1, facecolor="#D9D9D9", edgecolor='black', label='Other')
    ]
    
    # Structure legend
    structure_legend_handles = []
    structure_legend_labels = []
    unique_segments = sorted(set(bound_segments))
    
    for segment in unique_segments:
        color = segment_colors.get(segment, "#D9D9D9")
        structure_legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
        structure_legend_labels.append(f'Section {segment}')
    
    # Place legends side by side below the last subplot
    chord_legend = axes[-1, 0].legend(
        chord_legend_handles, 
        [h.get_label() for h in chord_legend_handles],
        loc='upper left', 
        bbox_to_anchor=(0.0, -0.15),
        ncol=5,
        frameon=True, 
        fontsize=10,
        title="Chord Types"
    )
    
    structure_legend = axes[-1, 0].legend(
        structure_legend_handles, 
        structure_legend_labels,
        loc='upper right', 
        bbox_to_anchor=(1.0, -0.15),
        ncol=min(4, len(structure_legend_handles)),
        frameon=True, 
        fontsize=10,
        title="Sections"
    )
    
    # Add the chord legend back (matplotlib removes it when adding the second one)
    axes[-1, 0].add_artist(chord_legend)
    
    plt.tight_layout()
    return fig


def blue_gradient(n, start="#004a7f", end="#009DFF"):
    """Return a list of n colors interpolated from start to end."""
    return [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
        "blue_grad", [start, end]
    )(np.linspace(0, 1, n))]

def plot_bass_by_beat_from_results(results, instrument_name='bass', start_beat=0, num_beats=16, figsize=(15, 20)):
    import matplotlib.pyplot as plt
    from music21 import pitch
    
    # Get beat_notes dict
    beat_notes = results['beat_notes']
    total_beats = len(beat_notes)
    end_beat = min(start_beat + num_beats, total_beats)
    actual_num_beats = end_beat - start_beat
    
    beats_per_row = 8
    cols = min(beats_per_row, actual_num_beats)
    rows = (actual_num_beats + cols - 1) // cols
    
    fig = plt.figure(figsize=figsize)
    
    # Create a mapping from note names to colors using blue gradient
    note_names = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Bb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']
    gradient_colors = blue_gradient(len(note_names))
    note_colors = dict(zip(note_names, gradient_colors))
    
    for i, beat in enumerate(range(start_beat, end_beat)):
        ax = fig.add_subplot(rows, cols, i + 1)
        notes = beat_notes.get(beat, [])
        midi_notes = [n for n in notes if n['instrument'] == instrument_name]
        
        if not midi_notes:
            ax.text(0.5, 0.5, "No bass notes", ha='center', va='center', fontsize=14)
            ax.set_yticks([])
            ax.set_xticks([])
            continue
        
        # Sort and plot
        midi_notes.sort(key=lambda n: -n['pitch'])
        y_positions = np.arange(len(midi_notes))
        
        for j, note in enumerate(midi_notes):
            note_pitch = pitch.Pitch(note['pitch'])
            note_name = note_pitch.name
            
            # Handle both natural notes (C, D, E, etc.) and accidentals (C#, D#, etc.)
            color = note_colors.get(note_name, note_colors.get(note_name[0], gradient_colors[-1]))
            
            ax.barh(y_positions[j], width=0.8, left=0.1, height=0.7, color=color, alpha=0.7)
            ax.text(0.5, y_positions[j], f"{note_name}", va='center', ha='center', fontsize=12, color='white')
        
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"Beat {beat + 1}", fontsize=14)
    
    plt.tight_layout()
    plt.show()