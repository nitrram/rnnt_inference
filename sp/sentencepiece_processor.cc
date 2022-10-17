// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "sentencepiece_processor.h"

#include <map>
#include <set>
#include <utility>

#include "common.h"
#include "filesystem.h"
#include "model_factory.h"
#include "model_interface.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/strings/strip.h"
#include "util.h"

namespace sentencepiece {
namespace {

// Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK).
const char kSpaceSymbol[] = "\xe2\x96\x81";

// Encodes <unk> into U+2047 (DOUBLE QUESTION MARK),
// since this character can be useful both for user and
// developer. We can easily figure out that <unk> is emitted.
const char kDefaultUnknownSymbol[] = " \xE2\x81\x87 ";

// REPLACEMENT CHARACTER (U+FFFD) in UTF-8.
const char kReplacementCharacter[] = "\xef\xbf\xbd";

std::vector<absl::string_view> ToPieceArray(const std::vector<std::string> &v) {
  std::vector<absl::string_view> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = v[i];
  return out;
}

void ConvertToUnicodeSpansInternal(SentencePieceText *spt) {
  if (spt == nullptr || spt->text().empty()) return;

  std::vector<int> utf8_to_unicode(spt->text().size() + 1, 0);
  absl::string_view str = spt->text();
  size_t prev = 0;
  int ulen = 0;
  while (!str.empty()) {
    const size_t mblen = std::max<int>(1, string_util::OneCharLen(str.data()));
    for (size_t i = prev; i < prev + mblen; ++i) {
      utf8_to_unicode[i] = ulen;
    }
    ++ulen;
    prev += mblen;
    str.remove_prefix(mblen);
  }
  utf8_to_unicode[prev] = ulen;

  auto clip = [&](int s) {
    return std::min<int>(std::max<int>(0, s), utf8_to_unicode.size() - 1);
  };

  for (auto &piece : *(spt->mutable_pieces())) {
    piece.set_begin(utf8_to_unicode[clip(piece.begin())]);
    piece.set_end(utf8_to_unicode[clip(piece.end())]);
  }
}

}  // namespace

ImmutableSentencePieceText::ImmutableSentencePieceText()
    : spt_(&SentencePieceText::default_instance()) {}

ImmutableSentencePieceText::ImmutableSentencePieceText(
    const SentencePieceText &spt)
    : spt_(&spt) {}

ImmutableSentencePieceText::~ImmutableSentencePieceText() {}

ImmutableSentencePieceText_ImmutableSentencePiece::
    ImmutableSentencePieceText_ImmutableSentencePiece()
    : sp_(&SentencePieceText_SentencePiece::default_instance()) {}

ImmutableSentencePieceText_ImmutableSentencePiece::
    ImmutableSentencePieceText_ImmutableSentencePiece(
        const SentencePieceText_SentencePiece &sp)
    : sp_(&sp) {}

const std::string &ImmutableSentencePieceText_ImmutableSentencePiece::piece()
    const {
  return sp_->piece();
}

const std::string &ImmutableSentencePieceText_ImmutableSentencePiece::surface()
    const {
  return sp_->surface();
}

uint32_t ImmutableSentencePieceText_ImmutableSentencePiece::id() const {
  return sp_->id();
}

uint32_t ImmutableSentencePieceText_ImmutableSentencePiece::begin() const {
  return sp_->begin();
}

uint32_t ImmutableSentencePieceText_ImmutableSentencePiece::end() const {
  return sp_->end();
}

std::vector<ImmutableSentencePieceText_ImmutableSentencePiece>
ImmutableSentencePieceText::pieces() const {
  std::vector<ImmutableSentencePieceText_ImmutableSentencePiece> pieces(
      spt_->pieces_size());
  for (int i = 0; i < spt_->pieces_size(); ++i)
    pieces[i] =
        ImmutableSentencePieceText_ImmutableSentencePiece(spt_->pieces(i));
  return pieces;
}

size_t ImmutableSentencePieceText::pieces_size() const {
  return spt_->pieces_size();
}

ImmutableSentencePieceText_ImmutableSentencePiece
ImmutableSentencePieceText::pieces(int index) const {
  return ImmutableSentencePieceText_ImmutableSentencePiece(spt_->pieces(index));
}

const std::string &ImmutableSentencePieceText::text() const {
  return spt_->text();
}

float ImmutableSentencePieceText::score() const {
  return spt_ ? spt_->score() : 0.0;
}

SentencePieceText *ImmutableSentencePieceText::mutable_proto() {
  if (rep_ == nullptr) {
    rep_ = std::make_shared<SentencePieceText>();
    spt_ = rep_.get();
  }
  return rep_.get();
}

void ImmutableSentencePieceText::ConvertToUnicodeSpans() {
  ConvertToUnicodeSpansInternal(mutable_proto());
}

util::bytes ImmutableSentencePieceText::SerializeAsString() const {
  return spt_->SerializeAsString();
}

ImmutableNBestSentencePieceText::ImmutableNBestSentencePieceText() {}
ImmutableNBestSentencePieceText::~ImmutableNBestSentencePieceText() {}

size_t ImmutableNBestSentencePieceText::nbests_size() const {
  return rep_ ? rep_->nbests_size() : 0;
}

ImmutableSentencePieceText ImmutableNBestSentencePieceText::nbests(
    int index) const {
  return ImmutableSentencePieceText(rep_->nbests(index));
}

std::vector<ImmutableSentencePieceText>
ImmutableNBestSentencePieceText::nbests() const {
  if (rep_ == nullptr) return {};
  std::vector<ImmutableSentencePieceText> nbests(rep_->nbests_size());
  for (int i = 0; i < rep_->nbests_size(); ++i)
    nbests[i] = ImmutableSentencePieceText(rep_->nbests(i));
  return nbests;
}

NBestSentencePieceText *ImmutableNBestSentencePieceText::mutable_proto() {
  if (rep_ == nullptr) {
    rep_ = std::make_shared<NBestSentencePieceText>();
  }
  return rep_.get();
}

void ImmutableNBestSentencePieceText::ConvertToUnicodeSpans() {
  if (!mutable_proto()) return;
  for (auto &spt : *(mutable_proto()->mutable_nbests())) {
    ConvertToUnicodeSpansInternal(&spt);
  }
}

util::bytes ImmutableNBestSentencePieceText::SerializeAsString() const {
  return rep_ ? rep_->SerializeAsString() : "";
}

SentencePieceProcessor::SentencePieceProcessor() {}
SentencePieceProcessor::~SentencePieceProcessor() {}

util::Status SentencePieceProcessor::Load(absl::string_view filename) {
  auto model_proto = absl::make_unique<ModelProto>();
  RETURN_IF_ERROR(io::LoadModelProto(filename, model_proto.get()));
  return Load(std::move(model_proto));
}

void SentencePieceProcessor::LoadOrDie(absl::string_view filename) {
  CHECK_OK(Load(filename));
}

util::Status SentencePieceProcessor::Load(const ModelProto &model_proto) {
  auto model_proto_copy = absl::make_unique<ModelProto>();
  *model_proto_copy = model_proto;
  return Load(std::move(model_proto_copy));
}

util::Status SentencePieceProcessor::LoadFromSerializedProto(
    absl::string_view serialized) {
  auto model_proto = absl::make_unique<ModelProto>();
  CHECK_OR_RETURN(
      model_proto->ParseFromArray(serialized.data(), serialized.size()));
  return Load(std::move(model_proto));
}

util::Status SentencePieceProcessor::Load(
    std::unique_ptr<ModelProto> model_proto) {
  model_proto_ = std::move(model_proto);
  model_ = ModelFactory::Create(*model_proto_);
  normalizer_ = absl::make_unique<normalizer::Normalizer>(
      model_proto_->normalizer_spec(), model_proto_->trainer_spec());
  if (model_proto_->has_denormalizer_spec() &&
      !model_proto_->denormalizer_spec().precompiled_charsmap().empty()) {
    denormalizer_ = absl::make_unique<normalizer::Normalizer>(
        model_proto_->denormalizer_spec());
  }

  // Escapes user-defined-symbols in normalizer.
  normalizer_->SetPrefixMatcher(model_->prefix_matcher());

  RETURN_IF_ERROR(status());

  return util::OkStatus();
}

util::Status SentencePieceProcessor::SetDecodeExtraOptions(
    absl::string_view extra_options) {
  return ParseExtraOptions(extra_options, &decode_extra_options_);
}

util::Status SentencePieceProcessor::status() const {
  CHECK_OR_RETURN(model_) << "Model is not initialized.";
  CHECK_OR_RETURN(normalizer_) << "Normalizer is not initialized.";
  RETURN_IF_ERROR(model_->status());
  RETURN_IF_ERROR(normalizer_->status());
  return util::OkStatus();
}

util::Status SentencePieceProcessor::SetVocabulary(
    const std::vector<absl::string_view> &valid_vocab) {
  RETURN_IF_ERROR(status());

  // TODO(taku): supports vocabulary constraint in BPE model.
  const auto type = model_proto_->trainer_spec().model_type();
  CHECK_OR_RETURN(type == TrainerSpec::UNIGRAM || type == TrainerSpec::BPE)
      << "Vocabulary constraint is only enabled in subword units.";

  const std::set<absl::string_view> vocab(valid_vocab.begin(),
                                          valid_vocab.end());

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    auto *piece = model_proto_->mutable_pieces(i);
    if (piece->type() == ModelProto::SentencePiece::CONTROL ||
        piece->type() == ModelProto::SentencePiece::UNKNOWN ||
        piece->type() == ModelProto::SentencePiece::USER_DEFINED) {
      continue;
    }
    if (vocab.find(piece->piece()) != vocab.end() ||
        string_util::OneCharLen(piece->piece().c_str()) ==
            piece->piece().size()) {
      piece->set_type(ModelProto::SentencePiece::NORMAL);
    } else {
      piece->set_type(ModelProto::SentencePiece::UNUSED);
    }
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::ResetVocabulary() {
  RETURN_IF_ERROR(status());
  for (auto &piece : *(model_proto_->mutable_pieces())) {
    if (piece.type() == ModelProto::SentencePiece::UNUSED)
      piece.set_type(ModelProto::SentencePiece::NORMAL);
  }

  return util::OkStatus();
}

#define CHECK_OR_RETURN_STATUS_STL(container)               \
  RETURN_IF_ERROR(status());                                \
  CHECK_OR_RETURN(container) << "output container is null"; \
  container->clear();

#define CHECK_OR_RETURN_STATUS_PROTO(proto)         \
  RETURN_IF_ERROR(status());                        \
  CHECK_OR_RETURN(proto) << "output proto is null"; \
  proto->Clear();

//////////////////////////////////////////////////////////////
// Simple API.

util::Status SentencePieceProcessor::Decode(
    const std::vector<std::string> &pieces, std::string *detokenized) const {
  return Decode(ToPieceArray(pieces), detokenized);
}

util::Status SentencePieceProcessor::Decode(
    const std::vector<absl::string_view> &pieces,
    std::string *detokenized) const {
  CHECK_OR_RETURN_STATUS_STL(detokenized);

  SentencePieceText spt;
  RETURN_IF_ERROR(Decode(pieces, &spt));
  *detokenized = std::move(spt.text());

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                            std::string *detokenized) const {
  CHECK_OR_RETURN_STATUS_STL(detokenized);

  SentencePieceText spt;
  RETURN_IF_ERROR(Decode(ids, &spt));
  *detokenized = std::move(spt.text());

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(
    const std::vector<std::string> &pieces, SentencePieceText *spt) const {
  return Decode(ToPieceArray(pieces), spt);
}

util::Status SentencePieceProcessor::Decode(
    const std::vector<absl::string_view> &pieces,
    SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  const char *unk_surface = kDefaultUnknownSymbol;
  if (model_proto_ && model_proto_->trainer_spec().has_unk_surface())
    unk_surface = model_proto_->trainer_spec().unk_surface().c_str();

  // Returns decoded piece and a boolean indicating if the function has consumed
  // a bos whitespace token (a piece starting with a kSpaceSymbol). This is used
  // to strip only the first whitespace token from the decoded sequence for
  // add_dummy_prefix.
  auto DecodeSentencePiece =
      [&](absl::string_view piece, int id,
          bool is_bos_ws) -> std::pair<std::string, bool> {
    if (IsControl(id)) {                 // <s>, </s>
      return std::make_pair("", false);  // invisible symbol.
    } else if (IsUnknown(id)) {
      if (IdToPiece(id) == piece) {  // <unk>
        return std::make_pair(unk_surface, false);
      } else {  // return piece when piece is not <unk>.
        return std::make_pair(std::string(piece), false);
      }
    }

    bool has_bos_ws = false;  // whether the token starts with a kSpaceSymbol
    if (is_bos_ws &&
        (!model_proto_ ||
         (model_proto_ &&
          (model_proto_->normalizer_spec().add_dummy_prefix() ||
           model_proto_->normalizer_spec().remove_extra_whitespaces())))) {
      // Consume if the current position is bos and
      // piece starts with kSpaceSymbol.
      has_bos_ws = absl::ConsumePrefix(&piece, kSpaceSymbol);

      if (model_proto_ &&
          model_proto_->normalizer_spec().remove_extra_whitespaces()) {
        // if we are removing extra whitespace, we remove all leading whitespace
        has_bos_ws = false;
      }
    }

    return std::make_pair(absl::StrReplaceAll(piece, {{kSpaceSymbol, " "}}),
                          has_bos_ws);
  };

  for (absl::string_view w : pieces) {
    auto *sp = spt->add_pieces();
    sp->mutable_piece()->assign(w.data(), w.size());
    sp->set_id(PieceToId(w));
  }

  RETURN_IF_ERROR(ApplyExtraOptions(decode_extra_options_, spt));

  std::string *text = spt->mutable_text();
  auto SetSurface = [&](int index, absl::string_view surface) {
    auto *sp = spt->mutable_pieces(index);
    sp->set_surface(std::string(surface));
    sp->set_begin(text->size());
    sp->set_end(text->size() + surface.size());
    absl::StrAppend(text, surface);
  };

  auto ProcessBytePieces = [&](int token_index_begin,
                               int token_index_end) -> util::Status {
    if (token_index_begin >= token_index_end) {
      return util::OkStatus();
    }

    // Constructs byte sequence.
    std::string bytes;
    for (int i = token_index_begin; i < token_index_end; ++i) {
      const auto &sp = spt->pieces(i);
      const int byte = PieceToByte(sp.piece());
      CHECK_LE_OR_RETURN(0, byte);
      bytes.append(1, byte);
    }

    // Set surfaces of `bytes` for each Unicode character.
    int offset = 0;
    const int bytes_len = bytes.size();
    while (offset < bytes_len) {
      // Consume `bytes` by one Unicode character.
      size_t consumed;  // Number of bytes consumed in this iteration.
      const bool is_valid = string_util::IsValidDecodeUTF8(
          absl::string_view(bytes).substr(offset), &consumed);

      // Set surfaces of the consumed byte pieces.
      const int token_index = token_index_begin + offset;

      if (!is_valid) {
        // The byte piece at `token_index` is structurally invalid. Map it to
        // REPLACEMENT CHARACTER (U+FFFD).
        CHECK_EQ_OR_RETURN(consumed, 1);
        SetSurface(token_index, kReplacementCharacter);
      } else {
        const absl::string_view utf8 =
            absl::string_view(bytes).substr(offset, consumed);
        for (size_t j = 0; j < consumed; j++) {
          // The last byte piece holds the surface of the original unknown
          // character. The other byte pieces hold an empty string as
          // surface.
          if (j == consumed - 1) {
            SetSurface(token_index + j, utf8);
          } else {
            SetSurface(token_index + j, "");
          }
        }
      }
      offset += consumed;
    }
    CHECK_EQ_OR_RETURN(token_index_begin + offset, token_index_end);

    return util::OkStatus();
  };

  int byte_start = 0;
  bool is_bos_ws = true;  // whether we expect a bos ws token to consume.
  bool bos_ws_seen = false;
  std::string decoded;

  for (int i = 0; i < spt->pieces_size(); ++i) {
    const auto &sp = spt->pieces(i);
    if (!IsByte(sp.id())) {
      RETURN_IF_ERROR(ProcessBytePieces(byte_start, i));

      // if we have seen a bos_ws token or any non-empty token
      if (bos_ws_seen || !text->empty()) is_bos_ws = false;

      byte_start = i + 1;
      std::tie(decoded, bos_ws_seen) =
          DecodeSentencePiece(sp.piece(), sp.id(), is_bos_ws);

      SetSurface(i, decoded);
    }
  }
  RETURN_IF_ERROR(ProcessBytePieces(byte_start, spt->pieces_size()));

  if (denormalizer_) {
    *text = denormalizer_->Normalize(*text);
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                            SentencePieceText *spt) const {
  std::vector<std::string> pieces;
  const int num_pieces = GetPieceSize();
  pieces.reserve(ids.size());
  for (const int id : ids) {
    if (id < 0 || id >= num_pieces) {
      return util::Status(util::StatusCode::kOutOfRange,
                          absl::StrCat("Invalid id: ", id));
    }
    pieces.emplace_back(IdToPiece(id));
  }
  return Decode(pieces, spt);
}

#define CHECK_STATUS_OR_RETURN_DEFAULT(value)                                \
  if (!status().ok()) {                                                      \
    LOG(ERROR) << status().message() << "\nReturns default value " << value; \
    return value;                                                            \
  }

int SentencePieceProcessor::GetPieceSize() const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->GetPieceSize();
}

int SentencePieceProcessor::PieceToId(absl::string_view piece) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->PieceToId(piece);
}

const std::string &SentencePieceProcessor::IdToPiece(int id) const {
  static const std::string *kEmptyString = new std::string;
  CHECK_STATUS_OR_RETURN_DEFAULT(*kEmptyString);
  return model_->IdToPiece(id);
}

float SentencePieceProcessor::GetScore(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0.0);
  return model_->GetScore(id);
}

bool SentencePieceProcessor::IsControl(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->IsControl(id);
}

bool SentencePieceProcessor::IsUnknown(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->IsUnknown(id);
}

bool SentencePieceProcessor::IsUnused(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(false);
  return model_->IsUnused(id);
}

bool SentencePieceProcessor::IsByte(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(false);
  return model_->IsByte(id);
}

int SentencePieceProcessor::unk_id() const {
  const int id = PieceToId(absl::string_view(model_->unk_piece().data()));
  if (IsUnknown(id)) return id;
  return -1;
}

int SentencePieceProcessor::bos_id() const {
  const int id = PieceToId(absl::string_view(model_->bos_piece().data()));
  if (IsControl(id)) return id;
  return -1;
}

int SentencePieceProcessor::eos_id() const {
  const int id = PieceToId(absl::string_view(model_->eos_piece().data()));
  if (IsControl(id)) return id;
  return -1;
}

int SentencePieceProcessor::pad_id() const {
  const int id = PieceToId(absl::string_view(model_->pad_piece().data()));
  if (IsControl(id)) return id;
  return -1;
}

// static
util::Status SentencePieceProcessor::ApplyExtraOptions(
    const std::vector<ExtraOption> &extra_options,
    SentencePieceText *spt) const {
  for (const auto &extra_option : extra_options) {
    switch (extra_option) {
      case REVERSE:
        std::reverse(spt->mutable_pieces()->begin(),
                     spt->mutable_pieces()->end());
        break;
      case EOS: {
        auto *piece = spt->add_pieces();
        piece->set_id(PieceToId(absl::string_view(model_->eos_piece().data())));
        piece->set_piece(model_->eos_piece().data(),
                         model_->eos_piece().size());
      } break;
      case BOS: {
        auto *array = spt->mutable_pieces();
        array->Add();
        for (int i = array->size() - 1; i > 0; --i) {
          array->SwapElements(i - 1, i);
        }
        auto *piece = array->Mutable(0);
        piece->set_id(PieceToId(absl::string_view(model_->bos_piece().data())));
        piece->set_piece(model_->bos_piece().data(),
                         model_->bos_piece().size());
      } break;
      case UNK_PIECE: {
        for (int i = 0; i < spt->pieces_size(); ++i) {
          auto *piece = spt->mutable_pieces(i);
          if (IsUnknown(piece->id())) {
            piece->set_piece(model_->unk_piece().data(),
                             model_->unk_piece().size());
          }
        }
      } break;
      default:
        return util::InternalError("unknown extra_option type.");
    }
  }

  return util::OkStatus();
}

// static
util::Status SentencePieceProcessor::ParseExtraOptions(
    absl::string_view _extra_option,
    std::vector<SentencePieceProcessor::ExtraOption> *extra_options) const {
  absl::string_view extra_option(_extra_option.data(), _extra_option.size());

  extra_options->clear();
  if (extra_option.empty()) return util::OkStatus();

  RETURN_IF_ERROR(status());

  static std::map<absl::string_view, SentencePieceProcessor::ExtraOption>
      extra_option_map = {{"bos", SentencePieceProcessor::BOS},
                          {"eos", SentencePieceProcessor::EOS},
                          {"reverse", SentencePieceProcessor::REVERSE},
                          {"unk", SentencePieceProcessor::UNK_PIECE},
                          {"unk_piece", SentencePieceProcessor::UNK_PIECE}};
  for (const auto &s : absl::StrSplit(extra_option, ":")) {
    const auto it = extra_option_map.find(s);
    CHECK_OR_RETURN(it != extra_option_map.end())
        << "option \"" << s << "\" is not available.";
    extra_options->push_back(it->second);

    if (it->second == SentencePieceProcessor::BOS) {
      CHECK_OR_RETURN(
          !IsUnknown(PieceToId(absl::string_view(model_->bos_piece().data()))))
          << "id for `" << model_->bos_piece() << "` is not defined.";
    }
    if (it->second == SentencePieceProcessor::EOS) {
      CHECK_OR_RETURN(
          !IsUnknown(PieceToId(absl::string_view(model_->eos_piece().data()))))
          << "id for `" << model_->eos_piece() << "` is not defined.";
    }
  }
  return util::OkStatus();
}

void SentencePieceProcessor::SetModel(std::unique_ptr<ModelInterface> &&model) {
  model_ = std::move(model);
}

void SentencePieceProcessor::SetNormalizer(
    std::unique_ptr<normalizer::Normalizer> &&normalizer) {
  normalizer_ = std::move(normalizer);
}

const ModelProto &SentencePieceProcessor::model_proto() const {
  return *model_proto_;
}

std::string SentencePieceProcessor::serialized_model_proto() const {
  return model_proto_ ? model_proto_->SerializeAsString() : "";
}

// Set seed value of random generator.
// Do not set static_cast<unique_int>(-1),
// as this seed is reserved for initializing from
// std::random_device.
void SetRandomGeneratorSeed(unsigned int seed);

namespace io {
util::Status LoadModelProto(absl::string_view filename,
                            ModelProto *model_proto) {
  if (filename.empty()) {
    return util::NotFoundError("model file path should not be empty.");
  }

  auto input = filesystem::NewReadableFile(filename, true);
  RETURN_IF_ERROR(input->status());
  std::string serialized;
  CHECK_OR_RETURN(input->ReadAll(&serialized));
  CHECK_OR_RETURN(
      model_proto->ParseFromArray(serialized.data(), serialized.size()));

  return util::OkStatus();
}

util::Status SaveModelProto(absl::string_view filename,
                            const ModelProto &model_proto) {
  if (filename.empty()) {
    return util::NotFoundError("model file path should not be empty.");
  }
  auto output = filesystem::NewWritableFile(filename, true);
  RETURN_IF_ERROR(output->status());
  CHECK_OR_RETURN(output->Write(model_proto.SerializeAsString()));

  return util::OkStatus();
}
}  // namespace io
}  // namespace sentencepiece