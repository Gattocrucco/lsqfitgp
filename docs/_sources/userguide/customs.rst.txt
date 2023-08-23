.. lsqfitgp/docs/customs.rst
..
.. Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
..
.. This file is part of lsqfitgp.
..
.. lsqfitgp is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, either version 3 of the License, or
.. (at your option) any later version.
..
.. lsqfitgp is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
..
.. You should have received a copy of the GNU General Public License
.. along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

.. currentmodule:: lsqfitgp

.. _customs:

A custom kernel: text classification
====================================

In the previous sections of the manual we always used the :class:`ExpQuad`
kernel. There are many other kernels available in the module; however, it will
be useful to build at least once our own kernel to understand more about how
kernels work.

First I will make up a problem for which there's not already a reasonable kernel
in the module. Let's say we want to recognize automatically the language of a
text. To keep things simple, we will use only two languages, English and Latin.
To represent this binary classification problem with a Gaussian process, we
consider a "score" which is modeled as a Gaussian process. Negative scores
correspond to English, positive scores correspond to Latin. Example: if the
prior for a text score is :math:`0 \pm 1`, it means there's 50 % probability
it's English and 50 % it's Latin. In general we have to integrate a Gaussian
distribution over the negative/positive semiaxis.

Without using statistical techniques, writing a language identification program
is quite simple; we could just put a list of common words for each language and
count the words in the text. However, for the sake of the example, we will
write a dumb kernel that has no internal knowledge of the languages and doesn't
even consider words.

How do we make a kernel over texts? We always talked about kernels as functions
of real numbers, but actually the definition did not require :math:`x` and
:math:`x'` to be numbers. It didn't require them to be *anything*. In
:ref:`kernelexpl` we wrote the positivity requirement with an integral, but it
can also be written as a sum:

.. math::
    \forall g: \sum_x \sum_{x'} g(x) k(x, x') g(x') \ge 0.

It still holds that a function of the form :math:`k(x, x') = \sum_i h_i(x)
h_i(x')` is a valid kernel. So if we invent a family of functions :math:`h_i`
on texts we are done. This kind of construction is called *mapping to feature
space*, each :math:`h_i` measures a "feature" of its input that matters for the
problem. The simplest functions we can define for texts are

.. math::
    h_i(x) = \text{how many times letter $i$ appears in text $x$}.

What does this mean practically? Fix :math:`i = \texttt{a}`. Then
:math:`h_\texttt{a}(x) h_\texttt{a}(x')` just multiplies the number of "a" in
two texts. If a text has no "a" at all, it will be zero-a-correlated with any
other text. Texts with a lot of "a" will be much a-correlated between them.

Let's put this into code::

    import lsqfitgp as lgp
    import numpy as np
    
    @lgp.kernel
    @np.vectorize
    def CountLetters(x, y):
        result = 0
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            xcount = 0
            for character in x:
                if character == letter:
                    xcount += 1
            ycount = 0
            for character in y:
                if character == letter:
                    ycount += 1
            result += xcount * ycount
        return result

We wrote a function that counts the occurences of a letter, multiplies the
counts, and then sums over all letters. We then applied two decorators:
``np.vectorize``, which applies the function to each text in ``x`` and ``y``
when they are arrays instead of single texts, and ``lgp.kernel``, which marks
the function as a kernel for :mod:`lsqfitgp`. The decorator order matters:
first we transform the function to make it work on arrays, and only then we
can apply ``@lgp.kernel`` to mark it as a kernel.

Note that this function is disastrously inefficient, I just wrote it to be as
clear as possible.

Now we need some texts. I'll copy-pasted the first paragraph of random articles
from the `English <https://en.wikipedia.org/wiki/Special:Random>`_ and `Latin
<https://la.wikipedia.org/wiki/Specialis:Pagina_fortuita>`_ wikipedias.

::

    english_texts = [
        """Kiy Island (Russian: Кий-остров) is an island in the Onega Bay of the White Sea, 8 km off-shore and 15 km from the town of Onega. The island stretches for 2 kilometres from north-west to south-east, but its width does not exceed 800 meters. The island includes wide and sandy beaches, piny woods rich with berries, and large rocks. It has a remarkably interesting architectural ensemble, and hosts the history of the Kiysky Krestny monastery. Thousands of tourists go and visit the island.[1]""",
        """Haakon Andreas Olsen (18 September 1923[citation needed] – 22 August 2010) was a Norwegian physicist.""",
        """Le Chant des chemins de fer (The Song of the Railways) is a cantata in B minor by Hector Berlioz for tenor solo, choir and orchestra composed in June 1846 on lyrics by Jules Janin and premiered June 14 1846 for the inauguration of the gare de Lille.""",
        """Omanaia (Māori: Ōmanaia) is a settlement in the Hokianga area of Northland, New Zealand.""",
        """Printed electronics is a set of printing methods used to create electrical devices on various substrates. Printing typically uses common printing equipment suitable for defining patterns on material, such as screen printing, flexography, gravure, offset lithography, and inkjet. By electronic industry standards, these are low cost processes. Electrically functional electronic or optical inks are deposited on the substrate, creating active or passive devices, such as thin film transistors; capacitors; coils; resistors. Printed electronics is expected to facilitate widespread, very low-cost, low-performance electronics for applications such as flexible displays, smart labels, decorative and animated posters, and active clothing that do not require high performance.[1]""",
        """The Battle of Blair's Landing was fought on April 12, 1864, in Red River Parish, Louisiana, as a part of the Red River Campaign of the American Civil War.""",
        """Bradley Godden (born 13 July 1969) is an Australian former professional rugby league footballer who played in the 1990s. He played for the Newcastle Knights, Hunter Mariners, and the Leeds Rhinos as a fullback, wing or centre.""",
        """Milecastle 22 (Portgate) was a milecastle of the Roman Hadrian's Wall. Its remains exist as a low, turf covered platform just east of the Portgate roundabout (junction of the A68 and B6318). The platform is 0.5 metres (1.6 ft) on the east side, reducing to only a parch mark on the west side.[1]""",
        """Mešeišta is a village in the Debarca Municipality, Macedonia. It sits on the border between the Debarca Municipality and Ohrid Municipality, but administratively belongs to the former. Its FIPS code was MK65.""",
        """Taubensee (Kössen/Unterwössen) is a lake of Tyrol, Austria.""",
        """Local elections were held in Scotland in May 1992, to elect members to all 53 district councils. It was the last local election held under the Local Government (Scotland) Act 1973, which had established the two-tier system of regions and districts. Regional and district councils were abolished in 1996, and replaced with 29 new mainland unitary authorities under the terms of the Local Government etc. (Scotland) Act 1994.""",
        """New Fork is a ghost town in Sublette County, Wyoming, United States, near Boulder. It was one of the earliest settlements in the upper Green River valley. New Fork was established in 1888 by John Vible and Louis Broderson, Danish immigrants who had arrived in the United States in 1884. They established a store along the Lander cut-off of the Oregon Trail. By 1908 a small town had grown around the store, and in 1910 Vible built a dance hall, called The Valhalla.""",
        """"Tell Me You're Mine" is a song written by Ronald L. Fredianelli and Dico Vasin and performed by The Gaylords. It reached #2 on the U.S. pop chart and #3 on Cashbox in 1953.[1]""",
        """In Greek mythology, Hyettus (Ancient Greek: Ὕηττος - Hyettos) was a native of Argos thought to have been the first man ever to have exacted vengeance over adultery: he reputedly killed Molurus, whom he had caught with his wife, and was sent into exile. King Orchomenus of Boeotia received him hospitably and assigned to him some land, where the village Hyettus was subsequently founded and named after him.[1][2]""",
        """Sanxiantai[1] (Amis: nuwalian; Chinese: 三仙台; pinyin: Sānxiāntái) is an area containing a beach and several islands located on the coast of Chenggong Township, Taitung County, Taiwan. The beach stretches for ten kilometers in length. It is situated at the 112-kilometer mark. A popular tourist attraction for its rocky coastal views, the area is well known for its long footbridge in the shape of a sea dragon that connects the coast to the largest island.[2] The name Sanxiantai means "three immortals platform", referring to the island with three large standing rocks.[3]""",
        """A number of steamships have been named Delphine""",
        """Animaux is a French television channel themed on the animal world.""",
        """Lin Li, FREng, CEng, FIET, FLIA, FCIRP (Chinese: 李林; pinyin: Lǐ Lín), is professor of laser engineering at the University of Manchester Institute of Science and Technology.""",
        """In various urban activities, a vault is any type of movement that involves overcoming an obstacle by jumping, leaping, climbing or diving over an obstacle while using their feet, hands or not touching it at all. Although Parkour doesn't involve the idea of set movements,[1] traceurs use similar ways of moving [2] to quickly and efficiently pass over obstacles.""",
        """Pocono Airlines was a regional airline operating out of Pocono Mountains Municipal Airport and Wilkes-Barre/Scranton International Airport, founded by Walter E. ("Wally") Hoffman Jr.""",
        """Strokkers is a Spanish children's animated series produced by Neptuno Films in 2002. It was created by Josep Viciana and designed by Roman Rybakiewicz (both of whom were also involved in production for another Netpuno Films series Connie the Cow).""",
        """Harrison Fraker, FAIA is a professor of Architecture and Urban Design, and the former Dean of the UC Berkeley College of Environmental Design.""",
        """"Did It in a Minute" is a song performed by American musical duo Hall & Oates. Written by member Daryl Hall with Sara and Janna Allen, the song was released as the third of four singles from their tenth studio album Private Eyes in March 1982. Daryl Hall performs lead vocals, while John Oates provides backing harmony vocals.""",
        """Chaerodrys is a genus of obese weevils (insects in the family Brachyceridae).""",
        """A photographic album, or photo album, is a series of photographic prints collected by an individual person or family in the form of a book.[1][2][3] Some book-form photo albums have compartments which the photos may be slipped into; other albums have heavy paper with an abrasive surface covered with clear plastic sheets, on which surface photos can be put.[4] Older style albums often were simply books of heavy paper on which photos could be glued to or attached to with adhesive corners or pages.[4]""",
    ]
    
    latin_texts = [
        """Colloretum[1] (-i, n.) (alia nomina: Colloredum Montis Albani) (Italiane: Colloredo di Monte Albano; Foroiuliensice: Colorêt di Montalban) est oppidum Italiae et municipium, in Regione Foro Iulii-Venetia Iulia et in Provincia Utinensi situm.""",
        """Lánzhōu,[1] seu fortasse Lanceu[2] (litteris Sinicis 兰州), est urbs Serica et caput provinciae Gansu.""",
        """Aizecourt-le-Bas est commune Francicum 58 incolarum (anno 2009) praefecturae Samara in regione Picardia.""",
        """4009 Drobyshevskij,[1] olim designationibus 1977 EN1, 1982 BP3, et 1984 SP5 agnitus, est asteroides systematis solaris nostri, asteroidibus Cinguli Principalis attributus. Astronomis terrestribus magnitudinem absolutam 12.50 monstrat. Die 13 Martii 1977 a Nicolao Stepanovich Chernykh, astronomo apud Observatorium Astrophysicum Crimaeae versato, repertus est.[2]""",
        """Vaslui (Dacoromanice Județ Vaslui) est unus quadraginta duorum (cum urbe Bucaresto) Romaniae circulorum. Urbs eiusdem nominis est caput huius circuli, cui 455550 incolarum (anno 2002) sunt.""",
        """Argentolium[1] (Francogallice Argenteuil) est urbs et commune Francicum 104'962 incolarum (anno 2012) praefecturae Valle Esiae in regione Insula Franciae.""",
        """Rimogne est commune Francicum 1'423 incolarum (anno 2010) praefecturae Arduennae in regione Campania et Arduenna.""",
        """Respublica Helvetica vocabatur civitas quam Helvetis invitis anno 1798 Francici Confoederationem Helveticam invasi e tredecim pagis constituerunt.""",
        """Hills Road Sixth Form College ("Collegium sextae classis ad Viam Collium") est coeducationalis schola publica Cantabrigiae in Anglia sita, quae solum in forma sexta plenum graduum AS et A curriculum circa 2100 iuvenibus annos ab 16 ad 18 habentibus ex regione circumdante, ac cursus varissimos circa quattuor milibus discipulorum omnium annorum ad tempus in ratione educationis adultorum in classibus interdiu nocteque habitis praebet.""",
        """Aucopolis[1][2] (Anglice Auckland, Maorice Tamaki Makaurau) est urbs maxima Novae Zelandiae. Urbi regionique metropolitanae sunt 1 454 300 incolarum, ac 32 centesimae incolarum Novae Zelandiae sunt. Aucopolis in parte septentrionali Insulae Septentrionalis sita est.""",
        """Vaprium[1] (-i, n.) (alia nomina: Vavris[2]) (Italiane: Vaprio d'Adda) est oppidum Italiae et municipium, circiter 9 000 incolarum, in regione Langobardia et in Urbe metropolitana Mediolanensi situm. Incolae Vaprienses appellantur.""",
        """Mirepoix (Gasconice Mirapeish) est commune 197 incolarum (anno 2008) Franciae praefecturae Aegirtii in regione Meridiano et Pyrenaeo.""",
        """Volsci[1] populus antiquus Italiae atque diu hostes Romanorum fuerunt.""",
        """Lantenay est commune Francicum 263 incolarum (anno 2012) praefecturae Indis in regione orientali Rhodano et Alpibus (a die 1 Ianuarii 2016 Arvernia Rhodano et Alpibus).""",
        """Étouy est commune Francicum 795 incolarum (anno 2012) praefecturae Esiae in regione Picardia.""",
        """Astropecten polyacanthus est species stellae marinae familiae astropectinidarum.""",
        """Statoniensis lacus[1][2] (alia nomina: Mezzani lacus) (Italiane: Lago di Mezzano) est parvus lacus Italiae, in Regione Latio, in Provincia Viterbiensi et in municipium Valentani situm, prope Lacum Volsiniensem.""",
        """Annus 11 a.C.n. e serie paginarum brevium de annis.""",
        """Anna Reagan (Anglice saepissime Nancy Reagan, nata Anne Frances Robbins Novi Eboraci in urbe die 6 Iulii 1921; domi mortua Bel Air in oppido prope Angelopolim Californiae die 6 Martii 2016[1][2]), alumna Collegii Smith ubi rebus scaenicis studuit, fuit actrix Americana atque ab anno 1952 uxor Ronaldi Reagan. Quo praeside Civitatum Foederatarum electo, Anna Reagan ab anno 1981 ad annum 1989, coniunx praesidis, titulum Anglicum First Lady gessit.""",
        """Saint-Martial-d'Albarède est commune Francicum 490 incolarum (anno 2011) praefecturae Duranii in regione Aquitania.""",
        """Béla Németh (sive Bela Nemeth) sive Adalbertus Németh, est poeta Latinus Hungaricus.""",
        """Franciscus Salmhofer (Vindobonae natus die 22 Ianuarii 1900; ibidem mortuus die 22 Septembris 1975) fuit poeta, compositor et concentus magister Austriacus, qui inter alia Wiener Staatsoper praefectus fuit.""",
        """Deversorium Astoria sive vulgo Hôtel Astoria est clarum historicum et lautissimum deversorium quinque stellarum Bruxellis Belgii situm, locus clarus in quem affluunt iam ante unum saeculum homines famosi, reges, scriptores et regentes mundi.""",
        """Concressault est commune Francicum 221 incolarum (anno 2011) praefecturae Cari in regione Media.""",
        """Clairfayts est commune 356 incolarum (anno 2008) praefecturae Septentrionis in Franciae borealis regione Septentrione et Freto.""",
    ]

Ok, so we have 25 english texts and 25 latin texts. There's some contamination,
like the latin wikipedia page on the *Hills Road Sixth Form College*, but this
is life. Let's put all this in a Gaussian process object::

    gp = (lgp
        .GP(CountLetters())
        .addx(english_texts, 'english')
        .addx(latin_texts, 'latin')
    )

I'm a bit worried. Will this mess really work? I'll look at the prior to see
if it makes sense before going on::

    prior = gp.prior()
    for label, texts in [('english', english_texts), ('latin', latin_texts)]:
        print('*****', label, '*****')
        for i in range(len(texts)):
            print('   ', texts[i][:20], '  ', prior[label][i])

Output:

.. code-block:: text

    ***** english *****
       Kiy Island (Russian:    0(95)
       Haakon Andreas Olsen    0(17)
       Le Chant des chemins    0(47)
       Omanaia (Māori: Ōman    0(20)
       Printed electronics     0(169)
       The Battle of Blair'    0(28)
       Bradley Godden (born    0(42)
       Milecastle 22 (Portg    0(55)
       Mešeišta is a villag    0(41)
       Taubensee (Kössen/Un    0(13)
       Local elections were    0(85)
       New Fork is a ghost     0(88)
       "Tell Me You're Mine    0(30)
       In Greek mythology,     0(81)
       Sanxiantai[1] (Amis:    0(117)
       A number of steamshi    0(12)
       Animaux is a French     0(15)
       Lin Li, FREng, CEng,    0(31)
       In various urban act    0(72)
       Pocono Airlines was     0(39)
       Strokkers is a Spani    0(48)
       Harrison Fraker, FAI    0(29)
       "Did It in a Minute"    0(61)
       Chaerodrys is a genu    0(17)
       A photographic album    0(98)
    ***** latin *****
       Colloretum[1] (-i, n    0(52)
       Lánzhōu,[1] seu fort    0(20)
       Aizecourt-le-Bas est    0(23)
       4009 Drobyshevskij,[    0(71)
       Vaslui (Dacoromanice    0(39)
       Argentolium[1] (Fran    0(32)
       Rimogne est commune     0(25)
       Respublica Helvetica    0(34)
       Hills Road Sixth For    0(92)
       Aucopolis[1][2] (Ang    0(55)
       Vaprium[1] (-i, n.)     0(48)
       Mirepoix (Gasconice     0(30)
       Volsci[1] populus an    0(16)
       Lantenay est commune    0(33)
       Étouy est commune Fr    0(20)
       Astropecten polyacan    0(21)
       Statoniensis lacus[1    0(46)
       Annus 11 a.C.n. e se    0(11)
       Anna Reagan (Anglice    0(87)
       Saint-Martial-d'Alba    0(25)
       Béla Németh (sive Be    0(17)
       Franciscus Salmhofer    0(42)
       Deversorium Astoria     0(55)
       Concressault est com    0(21)
       Clairfayts est commu    0(29)

Ok, what's this? The mean is zero because the prior mean is always zero. The
standard deviations are all different: should this be the case? Why should the
prior be particularly uncertain about the language score for the wikipedia
introduction to printed electronics? After all, if the mean is zero it means
50-50 whatever the standard deviation. Try to think about the answer, solution
in the footnote [#f2]_. Is this a problem? I don't know. Let's see what happens.

To use the Gaussian process, I need some other texts which I won't tell to it
if they are English or Latin. I'll pick this time::

    debellogallico = """
        Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae,
        aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli
        appellantur. Hi omnes lingua, institutis, legibus inter se differunt.
        Gallos ab Aquitanis Garumna flumen, a Belgis Matrona et Sequana
        dividit. Horum omnium fortissimi sunt Belgae, propterea quod a cultu
        atque humanitate provinciae longissime absunt, minimeque ad eos
        mercatores saepe commeant atque ea quae ad effeminandos animos
        pertinent important, proximique sunt Germanis, qui trans Rhenum
        incolunt, quibuscum continenter bellum gerunt. Qua de causa Helvetii
        quoque reliquos Gallos virtute praecedunt, quod fere cotidianis
        proeliis cum Germanis contendunt, cum aut suis finibus eos prohibent
        aut ipsi in eorum finibus bellum gerunt. Eorum una pars, quam Gallos
        obtinere dictum est, initium capit a flumine Rhodano, continetur
        Garumna flumine, Oceano, finibus Belgarum, attingit etiam ab Sequanis
        et Helvetiis flumen Rhenum, vergit ad septentriones. Belgae ab extremis
        Galliae finibus oriuntur, pertinent ad inferiorem partem fluminis
        Rheni, spectant in septentrionem et orientem solem. Aquitania a Garumna
        flumine ad Pyrenaeos montes et eam partem Oceani quae est ad Hispaniam
        pertinet; spectat inter occasum solis et septentriones.
    """
    
    paradiselost = """
        Of Man's first disobedience, and the fruit
        Of that forbidden tree whose mortal taste
        Brought death into the World, and all our woe,
        With loss of Eden, till one greater Man
        Restore us, and regain the blissful seat,
        Sing, Heavenly Muse, that, on the secret top
        Of Oreb, or of Sinai, didst inspire
        That shepherd who first taught the chosen seed
        In the beginning how the heavens and earth
        Rose out of Chaos: or, if Sion hill
        Delight thee more, and Siloa's brook that flowed
        Fast by the oracle of God, I thence
        Invoke thy aid to my adventurous song,
        That with no middle flight intends to soar
        Above th' Aonian mount, while it pursues
        Things unattempted yet in prose or rhyme.
    """
    
    gp = (gp
        .addx(debellogallico, 'caesar')
        .addx(paradiselost, 'milton')
    )

To say if a text is English or Latin, we have to give a specific score as data.
Short of better ideas, I'll use -1 and +1::

    post = gp.predfromdata({
        'english': -1 * np.ones(len(english_texts)),
        'latin': np.ones(len(latin_texts))
    }, ['caesar', 'milton'])

This time we passed a list of labels to :meth:`~GP.predfromdata`, which means
it returns a dictionary with the requested labels as keys. And the posterior
is::

    print(post['caesar'])
    print(post['milton'])

Output:

.. code-block:: text

    5.86042324(59)
    -2.4367508(15)

It works! The standard deviations are very small compared to the values, so the
probabilities of the sign are almost exactly 100 % (as a rule of thumb, a
Gaussian distribution has negligible mass farther than 5 standard deviations
from the mean).

Is it reasonable that it gets a completely determined result by just counting
the letters? In general different languages will tend to use more some letters.
Here the difference is stark, because in Latin there are no j, k, w, x, y at
all.

However, there's something that should bother you: the standard deviations are
not just small, they are *very* small. They're so small that they seem just an
artefact of the finite precision of floating point computation. Effectively, it
is so: the correct theorical value of those standard deviations is zero. The
reason is that we have fit 50 texts with a kernel that uses 26 letters. In
:ref:`kernelexpl` we said that fitting with a kernel :math:`\sum_i h_i(x)
h_i(x')` is equivalent to doing a linear least squares fit with parameters
:math:`p_i` where the model function is :math:`\sum_i p_i h_i(x)`, so we have
more data than parameters. This would not be a problem *if we had errors on the
datapoints*, but we put in the -1 and +1 without errors, so it's like solving
a system of 50 equations for 26 variables.

What are the output numbers then, if the fit is apparently not doable? In this
case :mod:`lsqfitgp` does something internally which is equivalent to adding a
very small uniform error to the datapoints. This is how the fit is solved, and
it is also why we get a very small but non-zero error on the output.

Now to another matter. In the end, it was not a problem that the prior
variances where depending on the length of the text. But still it feels wrong,
we initially stated the problem as determining if a text is English, not if it
is *a lot of* English. We set the datapoints to +1 and -1, Milton comes out as
roughly -2, while Caesar is +6. So Caesar is... very Latin?

We can fix this by normalizing the kernel such that the variance is always one.
It is like replacing a covariance matrix with a correlation matrix. We'll
use the :class:`Rescaling` kernel::

    kernel = CountLetters()
    inv_sdev = lambda x: 1 / np.sqrt(kernel(x, x))
    norm = lgp.Rescaling(stdfun=inv_sdev)
    gp = lgp.GP(kernel * norm)

So, we instantiated our :class:`CountLetters` kernel, wrote a function
``inv_sdev`` that computes the inverse of its standard deviation, and used this
function to rescale the kernel. Now let's run the fit::

    gp = (gp
        .addx(english_texts, 'english')
        .addx(latin_texts, 'latin')
        .addx(debellogallico, 'caesar')
        .addx(paradiselost, 'milton')
    )
    
    post = gp.predfromdata({
        'english': -1 * np.ones(len(english_texts)),
        'latin': np.ones(len(latin_texts))
    }, ['caesar', 'milton'])
    
    print(post)

Output:

.. code-block:: text

    {'caesar': array(0.729906033(12), dtype=object), 'milton': array(-1.267473517(25), dtype=object)}

This time both means are close to ±1.

.. note::

    The technique of using -1 and +1 to map a binary variable to continuous
    values for regression works fine in this simple example but is in general
    poor.

.. rubric:: Footnotes

.. [#f2] It's the length. The printed electronics paragraph is the longest,
         and in general the Latin ones were shorter.
