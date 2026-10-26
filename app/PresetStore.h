#pragma once

#include <QString>
#include <QStringList>
#include <optional>
#include <string>
#include <vector>

#include "stages/PresetLibrary.h"
#include "stages/StagePreset.h"

namespace dlssvid {

// The folder of presets for the GUI (PR-9): AppDataLocation/presets (%APPDATA%\dlssvid\dlssvid\presets - the person's documents,
// roaming with the profile), a thin layer over PresetLibrary with the Russian texts, the import and export and the one mark «для новых
// проектов» (QSettings presets/newProject: a file name in the folder, a pipeline preset or an NR preset).
class PresetStore {
public:
    static QString Dir();
    static void SetDir(const QString& dir);  // tests: away from the user's profile
    static PresetLibrary Library();
    static std::filesystem::path Path(const QString& file);  // QString -> path without the ANSI code page
    static QString Text(const std::filesystem::path& p);

    struct Listed {
        QString file, name, when;  // when: HumanWhen(created)
        Preset preset;
        bool slot = false;
        QString error;                     // a refused file: its first fatal issue in Russian
        std::vector<PresetIssue> notes;    // the notes of a read file (SlotValue of a slot ...)
    };
    struct Listing {
        std::vector<Listed> items;    // the person's presets of the target, by name (QCollator ru)
        std::vector<Listed> autoSlots;  // «Перед применением», «Перед сбросом» when they exist (not `slots`: a Qt keyword)
        std::vector<Listed> refused;  // «Не прочитаны»
        int hidden = 0;               // files past the first 500
    };
    // target: a stage with card presets ("nr") or kPipelineTarget
    static Listing List(const std::string& target);
    // the file of a preset of the scope with this name, the case ignored («мягкий» replaces «Мягкий»), over every file
    static std::optional<QString> FindByName(const QString& name, PresetScope scope);

    // Russian texts: one issue; the fatal ones or the notes of a list joined; what a preset leaves out (the save dialog)
    static QString IssueText(const PresetIssue& i, bool withStage = false);
    static QStringList IssueTexts(const std::vector<PresetIssue>& issues, bool fatal, bool withStage = false);
    static QString LeftOutText(const std::vector<LeftOut>& l, bool withStage);  // «Не войдут в пресет: …»; "" when nothing

    static QString NewProjectPreset();                   // the marked file's name in Dir(); "" = the built-in defaults
    static void SetNewProjectPreset(const QString& file);  // a file name or path in Dir(); "" clears the mark

    struct Result {
        bool ok = false;
        bool exists = false;  // import: the name is taken in the scope - ask, then again with `replacing` or `asCopy`
        QString copyName;     // exists: the name `asCopy` saves under («Аниме (3)» when «(2)» is taken too)
        QString file, name, message;
        QStringList notes, problems;
    };
    static Result Import(const std::string& target, const QString& from, const QString& replacing = {}, bool asCopy = false);
    static Result Export(const QString& file, const QString& to);
    static Result Remove(const QString& file);  // permanent; clears the new-projects mark when it was on this file
};

}  // namespace dlssvid
