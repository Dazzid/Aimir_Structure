Response to Reviewers

Response to Reviewers — Round 3


Manuscript: Data-Driven Analysis of Musical Form and Harmonic Structure in AI-Generated Popular Music: A Case Study with Suno and Udio


We thank the editor and the reviewers for this round of comments. One global change affects numbers throughout the manuscript. While removing the byte-pair-encoding machinery, as recommended, we rebuilt the trigram and tetragram datasets from scratch and broadened their coverage: the earlier construction was restricted to windows of distinct chords, and the rebuilt datasets also retain repetition-based progressions such as I-IV-I-IV, the most frequent tetragram of the human collection, and I-V-I-V. We recomputed every statistic, table, and figure from the rebuilt datasets. The arguments and observations are unchanged; the numbers are updated throughout, and we added new calculations in response to the reviewers' questions.

---


Editor's Decision (Mark Gotham)


Decision: revisions requested; may undergo further peer review prior to acceptance. The letter asks for a point-by-point response and for changes to be highlighted in the revised manuscript, and draws attention to "the editor's observation of issues from the first round that have not been addressed."


---
Editor Meta-Review


Comment.Recommend replacing all BPE explanation with simple tetragram, as that is what the current method is using.


Answer: Done. All byte-pair-encoding language has been removed from the manuscript. The harmonic analysis is now described as a parallel tetragram dataset (sliding window of four chords), and the section-label analysis as section-label trigrams/tetragrams, consistent with the released code.


---


Comment. Section 2.3: I am not sure whether it's necessary to mention accessible implementations in this detail. While this was needed at the initial version, as the authors used an original method, it does not seem important to mention that some ACE methods are not easily accessible.


Answer: We removed only the accessibility wording from the Section 2.3 (Automatic Chord Estimation) paragraph, leaving the substantive content intact. Specifically, we deleted the word "accessible" from the opening sentence, changed "provide accessible implementations of" to "implement," and cut the *Chordino* install-difficulty clause ("while still functional as a Vamp plugin, suffer from platform- and dependency-compatibility issues … cannot be integrated into modern Python pipelines"). The reasons we do not use *Chordino* are kept: it is deprecated ("no longer maintained") and its histograms fail to capture modal-interchange progressions (Collins et al. 2025). All technical content remains — the major/minor-triad limitation of baseline libraries, the Collins et al. finding, and the DECIBEL results (Odekerken et al. 2020) with the Isophonics figures and citation are all retained.


---


Comment. Also, the sentences of Line 209-213 sound a bit contradictory to me, which explains the limitation of RNA on audio, as the latter part of the manuscript is going to analyze chord results in Roman numeral format anyway.


Answer: We have revised our description of the pipeline to avoid the appearance of contradiction. The PARC benchmark limitation applies to *end-to-end* models that predict Roman numerals directly from audio— which is not what we do. We run ACE to obtain absolute chord labels and then convert those to Roman numerals, via global key estimation and music21 (Section 4.1). The weakness of direct audio-to-RNA models is therefore the *reason* for our two-step route, not an argument against our own results. We reworded the sentence to make this explicit: it now specifies that it is direct audio-to-RNA models that achieve modest accuracy, and adds that we instead derive Roman numerals symbolically from the ACE chord labels, separating chord recognition from key-relative interpretation.


---


Comment. Some of the bibliography issues commented by Reviewer 1 were not addressed. Lin et al 2023 ICCC, Park et al 2019 ISMIR, librosa zenodo, omissing pages in many ISMIR items


Answer: Corrected. Lin et al. (2023) is now an ICCC 2023 `@inproceedings` entry (pp. 64–73). Park et al. (2019, ISMIR, pp. 620–627), Peeters (2023, ISMIR, pp. 749–756), and Wu and Yang (2020, ISMIR, pp. 142–149) are now `@inproceedings` entries with full venue and page numbers. Pages were also added to every remaining ISMIR/TISMIR entry that lacked them: Buisson et al. (2022, pp. 591–597), Buisson et al. (2024, pp. 207–214), Poltronieri et al. (2025, pp. 492–500), and Nieto et al. (2020, TISMIR, pp. 246–263). Casini et al. now cites the published version (TISMIR, vol. 9(1), pp. 194–209, 2026). Marmoret et al. (2023) was already corrected to TISMIR (pp. 167–185), Waseem Akram et al. (2025) to IEEE TASLP, and librosa to the Zenodo software citation. Serrà is now spelled consistently across all entries.


---


Comment. The authors responded that they added a paragraph acknowledging the limitation of internal curation logic in Section 6, but I couldn't find it.


Answer: The paragraph now opens the Discussion (Section 6), rewritten in this version and merged with the selection-bias argument: the platforms' internal surfacing logic (Suno's *New Songs* playlist, Udio's search index) is undisclosed, so we report differences between the music each platform surfaces, not between unconstrained model outputs. The engagement medians (1 like, 4 plays for both collections; (lines 281–287) Section 3) still point to unfiltered samples, so this appears to be a statement of scope, not a concession that curation shaped the results.


---


Comment. Maybe the authors can consider moving L273 into Section 6 or new section about limitation.


Answer: Done. We moved the selection-bias argumentation (originally L273, in Section 3) into the Discussion (Section 6). Section 3 now reports only the engagement metrics and keeps the table and figure (Table 2, Figure 1) — a purely descriptive statement of medians, percentiles, and outliers. The interpretive argument — that the low, equal engagement medians are inconsistent with a popularity-filtered sample and instead indicate unfiltered acquisition, and that Udio's nominal 24-hour-play ranking does not act as a popularity filter at these magnitudes — now opens the Discussion, merged with the existing paragraph on the platforms' undisclosed internal curation logic. The two now read as one coherent limitations discussion, and the redundant cross-reference in the curation paragraph was removed.


---


Comment.L477: by Casini et al. (2025),


Answer: Fixed. The citation was a parenthetical `\citep` placed directly after "by", so it compiled as "by (Casini et al., 2025)". Changed to a textual `\citet`; it now reads "by Casini et al. (2026)". The year differs from the editor's quote because the entry now cites the published version of the paper (TISMIR, vol. 9(1), pp. 194–209, 2026) instead of the preprint.


---


Comment. L509: I'm not sure if I missed something, but Figure 3 shows that Suno's usage of those permutations is quite limited.


Answer: The reviewer is right: we never stated plainly what this figure 5 is. It is a palette map. Each collection is normalized to its own maximum within a style, so the figure shows which progressions a collection draws on, not how many times it uses them. Read as a quantity plot it makes Suno look modest, which is exactly the misreading our missing explanation invited. We have rewritten the caption and the surrounding text to say this directly: the figure is about the palette, not the amount; the amount, and Suno's over-representation, are reported in Figure 6 and Table 4. The clarified figure and text are in the revised manuscript. We also re-render the plot with the last version of the data.


---


Comment. L517: It says "computed as risk ratios against ... using the method described in Section 4.3", but I'm not sure the authors really meant it. Section 4.3 explains how to compose a trigram, which is already used in section 5.1. The computation of the risk ratio is introduced in the latter part of this section.


Answer: We agree, and correct it in both sections. Section 4.3 only builds the n-gram datasets; it contains no risk-ratio method, and the risk-ratio computation is defined later in this same section. The sentence was also self-contradictory, calling the figure both a relative-frequency plot and a risk-ratio plot. We rewrote it. The first paragraph now states only what the figure shows, the relative frequency of the canonical progressions in each collection, with no reference to Section 4.3. The term "risk ratio" is introduced only where its formula is defined and where Table 4 reports its values.


---


Comment. Also, Figure 6 does not look like it shows RRs, which must be larger than 1 for Suno. The tick of the y-axis is currently noted as relative frequency. Therefore, the first paragraph of this section can exclude RR, along with the mention of RR in the caption of Figure 6. Readers won't find any problem understanding Figure 6 as it is, without the concept of RR.


Answer: Correct, the figure plots relative frequency, not risk ratios. We fixed both places. The first paragraph no longer mentions risk ratios; it now states only that the figure shows the relative frequency of the canonical progressions in each collection. The caption no longer says the bars are proportions of risk ratios; it now states that the bars are the relative frequency of each progression within a collection. Risk ratio appears only later in the section, where its formula is defined and where Table 4 reports the values.


---


Comment. Section 6.1: I'm still not clear whether the section is about applying MIR tools to "AI-Generated Music". The introduction made the same research question, but as far as I understood, the current manuscript does not include specific considerations of whether it's AI-generated or not when applying MIR tools to the audio. The current contents are about applying MIR to large corpora, not AI-generated music.


Answer: Agreed. The section is about the reliability of MIR tools applied at corpus scale, not about anything specific to AI-generated music; these tools analyze audio that presents itself as music, whatever its source. We retitled it "Applying MIR Tools to Large Music Corpora," so the heading now matches the content. The questions specific to AI-generated music are RQ2 and RQ3; RQ1 is the prior methodological question of whether the tools are reliable enough to support that analysis across 60,014 tracks.


---


Comment. Also, the contents have some clear overlap with Section 2. The authors might reduce it if they want.


Answer: We have reduced this overlap by removing the two sentences in this section that only restated Section 2: the PARC benchmark result on direct audio-to-Roman-numeral accuracy, and the survey of ACE and MSA tool development with its citation list. Both are already presented in the Background. We kept the limitations that are specific to our pipeline and not covered in Section 2: key-estimation error propagation, the enharmonic-equivalence encoding, the analysis granularity, and the point that applying the same tools uniformly across all three collections cancels tool error from the comparison.


---


Comment. L858: I understand this was added to address the reviewer's concern, and I agree that the concern was reasonable, but I'm not sure whether inserting these sentences here makes the manuscript better. I worry that the narrative won't sound natural to readers (it somehow feels like the sentences are inserted only to address reviewers' concerns, rather than to make the narrative or logic clear/strong). This can be discussed in a limitation part, but maybe not in the middle of this conclusion. I'll leave it to the authors to decide.


Answer: Agreed. We cut the argumentative framing and kept only the part that reports information. What remains is the data-based statement, that Suno applies a single harmonic template across all stated genres, independent of user intent, which follows directly from the results. The evaluative and rhetorical material has been removed. The passage now states the finding rather than arguing a position, so it no longer reads as inserted to answer reviewers.


---


Comment. I think the comments on Jazz can also be moved to a limitation section.


Answer: We removed the Jazz observation from the results (Section 5.1) and moved it into the limitations discussion (Section 6.1), rephrased as a labeling limitation: style labels are user-supplied and inconsistent across platforms, and the `Jazz' tag on Suno and Udio largely denotes ambient instrumental music rather than standard jazz harmony, so any per-genre comparison under that label should be read with caution.


---


Additional Review (with editor comments)


Comment. §2


"The computational analysis of musical structure and harmony is a focus of MIR (Müller, 2015)."


Stub, one-sentence section. Either expand or integrate into §2.1
Also update Müller edition 2015 -> 2021


Answer: Expanded. The one-sentence stub is now a short introduction to the section that signposts the four strands of prior work it reviews: AI music generation and its evaluation, the two methods we apply at corpus scale (music structure analysis and automatic chord estimation), and prior corpus-scale harmonic analyses of popular music. The Müller citation is updated to the 2nd edition (2021).


---


Comment. §2.1.1


Avoid having a single division of §2.1.
This section lists previous studies mostly without comment on the choices made here. A narrative frame (e.g., we considered X, Y, and settled on Y for reasons Z) would help. §2.2 and §2.3 provide this kind of structure, culminating in clear conclusion statements. They make for a much more convincing read. There are still infelicities in the wording like "We selected it because it”.


Answer: (1) We removed the lone subsubsection: the "Evaluation of AI-Generated Music" heading is gone and its content now runs directly under §2.1, so there is no single division now. (2) The subsection already carries a frame and a conclusion, so we did not prepend a taxonomy sentence: it opens by stating our position, that we analyze the musical properties of the output at corpus scale against human-composed music, links the Jazz Transformer's trigram-irregularity metric (Wu et al., 2020) to our own method in the body, and closes with the motivation for a data-driven comparison. (3) We fixed the wording infelicity: "We selected it because it directly outputs… and achieves…" is now "We adopt it for two reasons: it directly outputs the section labels our analysis requires, and it reaches state-of-the-art performance on the Harmonix Set across all four tasks."


---


Comment. §2.4 Harmony Analysis of Corpora


Has this been added later? It does not fit well into the flow of the article. The within-section wording is also hard to follow. Even the title "Harmony Analysis of Corpora” suggests adding harmonic analysis to existing corpora as opposed to "Harmonic Analysis Corpora” which are corpora of harmonic analyses. Then there's mention of “a” corpus that complicated the single-vs-general commentary and even “melodic analysis” (which features in some relevant datasets but certainly not all). Much more clarity needed.


Answer: We retitled the section "Harmonic Analysis of Corpora," which removes the ambiguity. We rewrote the confusing sentence: the vague "a hand-annotated corpus" is now attributed by name (the rock corpus of Temperley and de Clercq, 2013), and the off-topic "melodic analysis" is dropped, since this study concerns harmony. The section keeps its closing point that no prior work offers a corpus-level harmonic and structural comparison between current AI platforms and human music, which is the thread the new Section 2 introduction now signposts, so the section connects to the rest of the background.


---


Comment. §3


Again the writing here is hard to follow. At first, there seems to be a methodological mismatch between selecting for popularity on the Udio side and having no such control on the Suno. Then there are some comments about addressing such bias that suggest roughly equal popularity (likes, shares) but that measurement seems to be captured shortly after upload, so not giving time for those values to accrue and/or plateau. Then there’s an assertion that the sample is representative. I would find this easier to parse with a sequence that leads with the high-level goal:


We aim to create a representative, unfiltered sample.
The most practical method for this across apps is X.
The pros and cons of this approach are Y.
This topic appears earlier too.


Answer: The substance here is the possible selection-bias mismatch between the platforms, and that is settled with data, not with a reordering. Section 3 states the collection method for all three corpora and reports the engagement metrics directly: Suno and Udio share a median of 1 like and 4 plays, and the 90th percentile reaches only 2 likes / 26 plays for Suno and 3 likes / 32 plays for Udio. Udio's nominal 24-hour play-count ranking therefore did not produce a popularity-concentrated sample: both collections sit at the same near-zero engagement floor despite the different nominal selection methods. Engagement is measured close to upload, which is exactly why these counts are low and is consistent with an unfiltered feed of fresh uploads rather than accrued popularity. The interpretation of this evidence for selection bias is consolidated in the Discussion (Section 6), following the editor's request to move that argumentation into the limitations. We did not prepend a goal or representativeness statement to Section 3: the engagement metrics carry that point empirically and the Discussion carries the interpretation, so such a preamble would only restate the Discussion without adding evidence.


---


Comment. §4.1


Krumhansl and Kessler is not ideal for pitch-class distribution in popular music as it’s more classical oriented (major-minor only, none of the pop modality). At least discuss, if not, swap out for more pop-oriented profiles (e.g., by Dominque Vuvan). A summary of these profiles is provided in the GitHub repo attached to When in Rome (TISMIR 2023). Editor: Maybe this can be integrated into one of the discussion sections, such as 6.1, where the authors discuss the limitations of the current approach?


Answer: The manuscript mis-stated the method; we have corrected it. We use no pitch-class profile at all, neither Krumhansl-Kessler nor the pop-oriented alternatives the reviewer suggests (Vuvan, When in Rome), so the classical-versus-pop objection does not apply. The global key is estimated per track from the ACE chord labels by duration-weighted diatonic template matching: each of the 12 major and 12 minor keys is scored by the time spent on chords whose root and quality fit its scale degrees, comparing pitch classes so the model's sharp-based spelling plays no role. Sections 4.1 and 6.1 now describe this, and 6.1 states its limits, since a single global-key estimate can be wrong under modulation, modal mixture, or relative-major/minor ambiguity. The estimator runs identically on all three collections, so any error is shared and cannot produce the between-collection differences we report.


---


Comment. Table 3


Table 3 would be better reported in terms of root pitch class and quality. Pitch with spelling to pitch class is lossy so:


in general, report on what model predicts (here, pitch class, not spelling); do not use pitch spelling if it's not been predicted for properly.
in this case, the sequence E flat – F – G is very common but the enharmonic equivalent reported here (D# - F - G) is only even seen in this kind of erroneous context. Editor: Maybe the authors can add a simple line on the caption that consonance-ACE assumes enharmonic equivalence, as also noted in the latter section.


Answer: The reviewer is right, and the example is in our own table: it shows a G-minor passage (G:min, F:maj, C:min) in which the ♭VI chord, E♭ major, is emitted as D♯:maj. consonance-ACE predicts a root pitch class and a quality and labels the root with a sharp by default, so "D♯" is not a spelling choice the model made; it is pitch class 3, which reads as E♭ in this context. We added the caption line the editor suggested: Table 3 now states that the labels are root pitch class plus quality under enharmonic equivalence (sharp by default), not spelled pitches; the D♯ rows in the example table are read accordingly as E♭ in context. We keep the raw labels rather than re-spelling them, because re-spelling would impose a spelling decision the model never makes; the caption tells the reader how to read them.


---


Comment. §4.3 Trigram Dataset Generation


Again, the example chosen is off-putting. The juxtaposition of Ab7 and C#7 spelt as such is unlikely. Pick a neutral example and/or use pitch class (or indeed key-relative labels as that’s the direction of travel). Editor: I guess this example is from the old method, as consonance-ACE always uses sharp (as far as I understood)


Answer: We keep the example, because it is an illustration of the sliding-window trigram construction, not a sample of model output, and the text says so directly: "the example uses absolute chord labels for illustrative clarity." In this case the Ab7 is a tritone substitution of D7, working as a bII7 of Gm. The four chords were chosen by hand to show the windowing mechanism on a clear sequence; their spelling is an illustrative choice, not a claim about what consonance-ACE emits. The editor's inference that the example is a leftover from the old method is therefore mistaken: it is a deliberate illustration, not a residue of any pipeline, so consonance-ACE's sharp convention does not apply to it.


---


Comment.Typo: Collins (2025) demonstrate[s]


Answer:Corrected.


---


Comment.§4.5
This method statement completely mis-represents the source. Richards 2017 rightly observes that:


Rotation equivalence is relevant (e.g., aFCG and CGaF are both common).
Wider re-ordering is not: the cycle strongly tends to appears in the given order.
As such, taking the cycle as an unordered set then is contrary to Richards, to musical intuition, and to clear corpus evidence of what equivalence are common.


In the unlikely event that you want to dispute this, you have a significant burden of proof to convince the reader. Show us the relative frequency of relevant professions (rotation equivalent in set, non-rotation equivalent in set, out of set) to make the case. I’m sure you’ll find the numbers defend the Richards side and contradict the current paper approach.


Also note that the step of “normalizing” to Roman numerals can actually obscure equivalences: specifically, transposition equivalence. This is especially relevant, given the not fully convincing key assignment method as discussed above.


Editor: I also read Richards 2017, and it is not clear what the authors wanted to say at L430. The reviewer wanted explanation or justification, but the revised paragraph is just a disclaimer. I'm also not sure whether the authors think (or argue that) their permutation of chord is equivalent to"rotation" in Richards paper. If I understood correctly, I guess the authors are also arguing that re-ordering changes the meaning of the progression. But the problem is that it's difficult to understand because the authors do not explain why they made this decision deliberately. Why not compare the Axis of Awesome progression strictly? In the current form, the readers might suspect that measuring the appearance of Axis of Awesome in a strict format did not make this clear difference, as it's difficult to guess what the motivation of using full perturbation is.


Answer: We ran the analysis suggested by the reviewer and find the relative frequencies of rotation-equivalent / non-rotation-in-set / out-of-set windows, computed over the four-chord windows of the deduplicated Roman numeral sequences, are: Lastfm 1.26% / 2.22% / 96.52%; Suno 6.73% / 5.67% / 87.60%; Udio 1.75% / 2.80% / 95.45%. Among windows built on an axis chord set, the share that is the literal loop (a rotation) is 54.3% for Suno against 36.3% for Lastfm and 38.5% for Udio. Suno is not "using the same four common chords": it plays the loop, in order, about 1.5 times more often than humans. We then applied the strict rotation-only definition (12 patterns, zero reordering). We find that Suno's lead over humans does not shrink, it grows at every threshold (Suno-to-Lastfm ratio of songs above the threshold, any-order → loop-only):
from songs using the Axis chords in more than the 10 percent of the song we move from 3.24× to 4.83×; more than the ≥25% from 4.54× to 6.22×; more than the ≥50%, from 8.07× to 12.48×. As an independent check, we recomputed the three-way classification from the released tetragram dataset, a separately constructed artifact and find no change.


The manuscript now reports both measures side by side. Section 4.4 defines axis-family membership (72 patterns) and the literal loop (12 rotations) and states the Richards rationale: the cycle strongly tends to appear in its given order. Table 5 reports the per-song rates under both measures, Figure 10 shows both distributions, and Figure 11 gives the loop-versus-reordering breakdown. On the Roman-numeral point: rotation equivalence also absorbs the relative-major/minor ambiguity of the global key estimate. C, G, Am, F is I-V-vi-IV when the key is read as C major and III-bVII-i-bVI, the minor-form set begun on a different chord, when it is read as A minor; treating the three sets and their rotations as one loop recovers the progression under either reading, so a track is not counted as non-axis merely because the key estimate landed on the relative key. Section 4.4 states this


We hope we have interpreted the reviewer comment correctly. 
